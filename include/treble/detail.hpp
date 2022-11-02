#ifndef TREBLE_DETAIL_H
#define TREBLE_DETAIL_H
//          Copyright Matthew Gibson 2022.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          https://www.boost.org/LICENSE_1_0.txt)

#include <array>
#include <type_traits>

namespace treble {
namespace detail {

/**
 * @brief Provides a way to count how many times a type T is included in a
 * parameter pack
 *
 * @tparam T Type to count
 */
template <typename T, typename First, typename... Rest>
struct type_count {
  constexpr static size_t value() noexcept {
    if constexpr (std::is_same_v<T, std::decay_t<First>>) {
      return 1 + type_count<T, Rest...>::value();
    } else {
      return type_count<T, Rest...>::value();
    }
  }
};

template <typename T, typename Last>
struct type_count<T, Last> {
  constexpr static size_t value() noexcept {
    if constexpr (std::is_same_v<T, std::decay_t<Last>>) {
      return 1;
    } else {
      return 0;
    }
  }
};

/**
 * @brief Simple way to concatenate two std::arrays of the same type
 */
template <typename T, size_t Size1, size_t Size2>
constexpr std::array<T, Size1 + Size2> array_cat(
    const std::array<T, Size1> &arr1,
    const std::array<T, Size2> &arr2) noexcept {
  std::array<T, Size1 + Size2> new_arr;
  for (size_t i = 0; i < Size1; ++i) {
    new_arr[i] = arr1[i];
  }
  for (size_t i = Size1; i < Size1 + Size2; ++i) {
    new_arr[i] = arr2[i - Size1];
  }
  return new_arr;
}

template <typename T, typename First, typename... Rest>
struct gather_array_from_tuple_impl {
  using array_type = std::array<T, type_count<T, First, Rest...>::value()>;
  constexpr static array_type exec(First &&first, Rest &&...rest) noexcept {
    if constexpr (std::is_same_v<T, std::decay_t<First>>) {
      return array_cat(std::array<T, 1>{first},
                       gather_array_from_tuple_impl<T, Rest...>::exec(
                           std::forward<Rest>(rest)...));
    } else {
      return gather_array_from_tuple_impl<T, Rest...>::exec(
          std::forward<Rest>(rest)...);
    }
  }
};

template <typename T, typename Last>
struct gather_array_from_tuple_impl<T, Last> {
  using array_type = std::array<T, type_count<T, Last>::value()>;
  constexpr static array_type exec(Last &&last) noexcept {
    if constexpr (std::is_same_v<T, std::decay_t<Last>>) {
      return {std::forward<Last>(last)};
    } else {
      return {};
    }
  }
};

template <typename T, typename... Ts, size_t... Seq>
constexpr decltype(auto) gather_array_from_tuple(
    std::index_sequence<Seq...>, std::tuple<Ts...> &tpl) noexcept {
  return gather_array_from_tuple_impl<
      T,
      decltype(std::get<std::integral_constant<size_t, Seq>::value>(tpl))...>::
      exec(std::get<std::integral_constant<size_t, Seq>::value>(tpl)...);
}

/**
 * @brief Makes a std::array containing in-order a copy of each occurance of
 * type T in a tuple
 */
template <typename T, typename... Ts>
constexpr decltype(auto) gather_array_from_tuple(
    std::tuple<Ts...> &tpl) noexcept {
  return gather_array_from_tuple<T, Ts...>(
      std::make_index_sequence<sizeof...(Ts)>{}, tpl);
}

template <typename T, size_t N, size_t arr_idx, size_t tpl_idx, typename... Ts>
struct scatter_array_to_tuple_impl {
  using tuple_type = std::tuple<Ts...>;
  constexpr static void exec(std::tuple<Ts...> &tpl,
                             std::array<T, N> &arr) noexcept {
    if constexpr (arr_idx < N && tpl_idx < std::tuple_size_v<tuple_type>) {
      if constexpr (std::is_same_v<T, std::decay_t<std::tuple_element_t<
                                          tpl_idx, tuple_type>>>) {
        std::get<tpl_idx>(tpl) = std::get<arr_idx>(arr);
        return scatter_array_to_tuple_impl<T, N, arr_idx + 1, tpl_idx + 1,
                                           Ts...>::exec(tpl, arr);
      } else {
        return scatter_array_to_tuple_impl<T, N, arr_idx, tpl_idx + 1,
                                           Ts...>::exec(tpl, arr);
      }
    }
  }
};

/**
 * @brief Copy an array of type T to the corresponding elements of the same
 * type stored in a tuple
 */
template <typename T, size_t N, typename... Ts>
constexpr static void scatter_array_to_tuple(std::tuple<Ts...> &tpl,
                                             std::array<T, N> &arr) noexcept {
  scatter_array_to_tuple_impl<T, N, 0, 0, Ts...>::exec(tpl, arr);
}

template <typename OptVar, typename... BoundArgs>
struct state_vector {
  using type = std::array<OptVar, type_count<OptVar, BoundArgs...>::value()>;
};
}  // namespace detail
}  // namespace treble

#endif TREBLE_DETAIL_H