// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"

namespace onnxruntime {
/**
 * A simple POD for using with tensor deserialization
 */
class MemBuffer {
 public:
  /**
   * @brief  A buffer with no ownership
  */
  MemBuffer(void* buffer, size_t len, const OrtMemoryInfo& alloc_info)
      : buffer_(buffer), len_(len), alloc_(nullptr), 
      alloc_info_(alloc_info) {}

  /**
   * @brief A buffer carrying ownership
  */
  MemBuffer(void* buffer, size_t len, AllocatorPtr&& alloc)
      : buffer_(buffer), len_(len), alloc_(std::move(alloc)),
      alloc_info_(alloc_->Info()) {}

  void* GetBuffer() const { return buffer_; }

  size_t GetLen() const { return len_; }

  const OrtMemoryInfo& GetAllocInfo() const { return alloc_info_; }

  bool IsOwner() const { return bool(alloc_); }

  AllocatorPtr Release() {
    AllocatorPtr ret(std::move(alloc_));
    alloc_.reset();
    return ret;
  }

  virtual ~MemBuffer() {
    if (nullptr != buffer_ && alloc_) {
      alloc_->Free(buffer_);
    }
  }
 private:
  void* const buffer_;
  const size_t len_;
  AllocatorPtr alloc_;
  const OrtMemoryInfo& alloc_info_;
};
};  // namespace onnxruntime
