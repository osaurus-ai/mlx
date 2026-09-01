// Copyright © 2023-2024 Apple Inc.
#include <memory>

#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"

namespace mlx::core::gpu {

void new_stream(Stream stream) {
  if (stream.device == mlx::core::Device::gpu) {
    metal::device(stream.device).new_queue(stream.index);
  }
}

// Routes a command-buffer failure through the installed mlx-c error handler
// (the same hook host apps install for `mlx_error`). Declared here because
// mlx core doesn't include mlx-c headers; both compile into the same binary.
// Linkage caveat: a build of this fork without mlx-c will fail to link this
// file — acceptable for the fork, whose only consumer (Cmlx in vmlx-swift)
// always compiles core and mlx-c together.
extern "C" void _mlx_error(const char* file, int line, const char* fmt, ...);

inline std::string error_message(MTL::CommandBuffer* cbuf) {
  std::ostringstream msg;
  msg << "[METAL] Command buffer execution failed: "
      << cbuf->error()->localizedDescription()->utf8String();
  return msg.str();
}

inline void check_error(MTL::CommandBuffer* cbuf) {
  if (cbuf->status() == MTL::CommandBufferStatusError) {
    throw std::runtime_error(error_message(cbuf));
  }
}

// Variant for MTLCommandBuffer completed handlers, which run on Metal's own
// dispatch queue (com.Metal.CompletionQueueDispatch). A C++ exception thrown
// there cannot unwind through the ObjC block / libdispatch frames, so `throw`
// becomes std::terminate -> abort and kills the whole process (e.g. a GPU
// command buffer failing under memory pressure while loading a large model).
// Report through the error handler instead; the default handler still exits,
// preserving upstream behavior for hosts that never install one.
// If a host installs a handler that returns, execution proceeds past the
// failed buffer and downstream arrays contain garbage — a recovering handler
// must abandon the in-flight generation, never log-and-continue it.
inline void check_error_in_completion_handler(MTL::CommandBuffer* cbuf) {
  if (cbuf->status() == MTL::CommandBufferStatusError) {
    _mlx_error(__FILE__, __LINE__, "%s", error_message(cbuf).c_str());
  }
}

void eval(array& arr) {
  auto pool = metal::new_scoped_memory_pool();
  auto s = arr.primitive().stream();
  auto& d = metal::device(s.device);
  auto command_buffer = d.get_command_buffer(s.index);

  auto outputs = arr.outputs();
  {
    // If the array is a tracer hold a reference
    // to its inputs so they don't get donated
    std::vector<array> inputs;
    if (arr.is_tracer()) {
      inputs = arr.inputs();
    }

    debug_set_primitive_buffer_label(command_buffer, arr.primitive());
    arr.primitive().eval_gpu(arr.inputs(), outputs);
  }
  std::unordered_set<std::shared_ptr<array::Data>> buffers;
  for (auto& in : arr.inputs()) {
    buffers.insert(in.data_shared_ptr());
  }
  for (auto& s : arr.siblings()) {
    buffers.insert(s.data_shared_ptr());
  }
  // Capture the output's data_shared_ptr so the output buffer's lifetime is
  // tied to CB completion. The unordered_set already dedupes the
  // input-donated-as-output case (same data_shared_ptr), so this is safe.
  buffers.insert(arr.data_shared_ptr());

  // Transfer encoder-side MTL::Buffer retains into the completion handler so
  // the Metal-level refcount survives until the GPU is done with the CB,
  // independent of any caller-side shared_ptr<array::Data> lifetime. Required
  // because allocator buffers use MTLResourceHazardTrackingModeUntracked and
  // the command buffer uses commandBufferWithUnretainedReferences().
  auto retained = d.take_retained_buffers(s.index);

  if (d.command_buffer_needs_commit(s.index)) {
    d.end_encoding(s.index);
    scheduler::notify_new_task(s);
    command_buffer->addCompletedHandler(
        [s, buffers = std::move(buffers), retained = std::move(retained)](
            MTL::CommandBuffer* cbuf) {
          scheduler::notify_task_completion(s);
          for (auto* b : retained) {
            if (b)
              b->release();
          }
          check_error_in_completion_handler(cbuf);
        });
    d.commit_command_buffer(s.index);
    d.get_command_buffer(s.index);
  } else {
    command_buffer->addCompletedHandler(
        [buffers = std::move(buffers),
         retained = std::move(retained)](MTL::CommandBuffer* cbuf) {
          for (auto* b : retained) {
            if (b)
              b->release();
          }
          check_error_in_completion_handler(cbuf);
        });
  }
}

void finalize(Stream s) {
  auto pool = metal::new_scoped_memory_pool();
  auto& d = metal::device(s.device);
  auto cb = d.get_command_buffer(s.index);
  d.end_encoding(s.index);
  cb->addCompletedHandler(
      [](MTL::CommandBuffer* cbuf) { check_error_in_completion_handler(cbuf); });
  d.commit_command_buffer(s.index);
  d.get_command_buffer(s.index);
}

void synchronize(Stream s) {
  auto pool = metal::new_scoped_memory_pool();
  auto& d = metal::device(s.device);
  auto cb = d.get_command_buffer(s.index);
  cb->retain();
  d.end_encoding(s.index);
  d.commit_command_buffer(s.index);
  cb->waitUntilCompleted();
  check_error(cb);
  cb->release();
}

} // namespace mlx::core::gpu
