#pragma once

#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "torch/csrc/lazy/core/int128.h"

namespace torch {
namespace lazy {

using size_t = std::size_t;
using hash_t = uint128;

hash_t HashBlock(const void* data, size_t n, const hash_t& seed);

hash_t DataHash(const void* data, size_t size);

size_t StdDataHash(const void* data, size_t size);

size_t StdHashCombine(uintmax_t a, uintmax_t b);

hash_t HashCombine(const hash_t& a, const hash_t& b);

size_t HashReduce(const hash_t& a);

std::string HexHash(const hash_t& a);

struct HashReducer {
  size_t operator()(const hash_t& value) const {
    return HashReduce(value);
  }
};

static inline hash_t StringHash(const char* data) {
  return DataHash(data, std::strlen(data));
}

template <
    typename T,
    typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
hash_t Hash(const T& value) {
  return DataHash(&value, sizeof(value));
}

static inline hash_t Hash(const std::string& value) {
  return DataHash(value.data(), value.size());
}

// Forward declare to allow hashes of vectors of vectors to work.
template <typename T>
hash_t ContainerHash(const T& values);

template <typename T>
hash_t Hash(const std::vector<T>& values) {
  return ContainerHash(values);
}

template <typename T>
hash_t Hash(const std::set<T>& values) {
  return ContainerHash(values);
}

template <typename T, typename S>
hash_t Hash(const std::pair<T, S>& values) {
  return HashCombine(Hash(values.first), Hash(values.second));
}

static inline hash_t Hash(const hash_t& value) {
  return value;
}

template <typename T>
hash_t ContainerHash(const T& values) {
  hash_t h = 0x85ebca77c2b2ae63;
  for (auto& value : values) {
    h = HashCombine(h, Hash(value));
  }
  return h;
}

template <typename T = void>
hash_t MHash() {
  return 0x165667b19e3779f9;
}

template <typename T, typename... Targs>
hash_t MHash(T value, Targs... Fargs) {
  return HashCombine(Hash(value), MHash(Fargs...));
}

} // namespace lazy
} // namespace torch
