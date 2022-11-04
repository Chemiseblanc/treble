#ifndef TREBLE_SELF_TUNING_FUNCTION_H
#define TREBLE_SELF_TUNING_FUNCTION_H
//          Copyright Matthew Gibson 2022.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          https://www.boost.org/LICENSE_1_0.txt)

#include <array>
#include <functional>
#include <type_traits>
#include <utility>

#include "treble/detail.hpp"
#include "treble/optimizers/incremental_sub_gradient.hpp"
#include "treble/param.hpp"

namespace treble {
namespace detail {
/**
 * @brief Uses SFINAE overload selection to turn a stored placeholder or tuning
 * parameter into the real value to be passed to a function
 *
 * @tparam T
 * @tparam PlaceholderArgs
 */
template <typename TuningParam, typename... PlaceholderArgs>
struct sub_placeholder_by_value {
  std::tuple<PlaceholderArgs &&...> args;

  explicit sub_placeholder_by_value(PlaceholderArgs &&...func_args) noexcept
      : args{std::forward<PlaceholderArgs>(func_args)...} {}

  template <typename U,
            std::enable_if_t<std::is_same_v<TuningParam, std::decay_t<U>>,
                             bool> = true>
  constexpr decltype(auto) operator[](U &tuning_param) noexcept {
    return tuning_param.value;
  }

  template <typename Placeholder,
            std::enable_if_t<std::is_placeholder_v<Placeholder>, bool> = true>
  constexpr decltype(auto) operator[](Placeholder) noexcept {
    return std::get<std::is_placeholder<Placeholder>::value - 1>(
        std::move(args));
  }
};
}  // namespace detail

/**
 * @brief Function object that creates a std::bind like interface for turning a
 * function into a self-tuning function for integer parameters
 */
template <typename Optimizer, typename BindPolicy, typename Callable,
          typename... Args>
class self_tuning_function {
 public:
  constexpr explicit self_tuning_function(Callable &&func,
                                          Args &&...args) noexcept
      : func{std::forward<Callable>(func)},
        arguments{std::forward<Args>(args)...} {}

  /**
   * @brief Calls the wrapped function, passing through arguments for the
   * placeholder values and return values. Measures the time for the function to
   * execute and passes it to the optimization entry-point
   */
  template <typename... FreeArgs>
  constexpr decltype(auto) operator()(FreeArgs &&...args) noexcept(
      noexcept(static_cast<BindPolicy *>(this)->call(
          std::make_index_sequence<sizeof...(Args)>{},
          std::forward<FreeArgs>(args)...))) {
    auto probe = optimizer.make_probe();
    return static_cast<BindPolicy *>(this)->call(
        std::make_index_sequence<sizeof...(Args)>{},
        std::forward<FreeArgs>(args)...);
  }

 protected:
  std::decay_t<Callable> func;
  std::tuple<std::decay_t<Args>...> arguments;

 private:
  Optimizer optimizer{arguments};
};

/**
 * @brief std::bind like interface for self_tuning_function
 */
template <typename Optimizer, typename Callable, typename... BoundArgs>
class st_fn_placeholders
    : public self_tuning_function<
          Optimizer, st_fn_placeholders<Optimizer, Callable, BoundArgs...>,
          Callable, BoundArgs...> {
 public:
  using parent_type = self_tuning_function<
      Optimizer, st_fn_placeholders<Optimizer, Callable, BoundArgs...>,
      Callable, BoundArgs...>;
  using parent_type::parent_type;

  friend parent_type;

 private:
  template <typename... FreeArgs, size_t... Seq>
  constexpr decltype(auto)
  call(std::index_sequence<Seq...>, FreeArgs &&...args) noexcept(
      noexcept(parent_type::func(
          (detail::sub_placeholder_by_value<tunable_param, FreeArgs...>{
              std::forward<FreeArgs>(args)...}[std::get<std::integral_constant<
              size_t, Seq>::value>(parent_type::arguments)])...))) {
    return parent_type::func(
        (detail::sub_placeholder_by_value<tunable_param, FreeArgs...>{
            std::forward<FreeArgs>(args)...}[std::get<std::integral_constant<
            size_t, Seq>::value>(parent_type::arguments)])...);
  }
};

/**
 * @brief std::bind_front like interface for self_tuning_function
 */
template <typename Optimizer, typename Callable, typename... BoundArgs>
class st_fn_front
    : public self_tuning_function<
          Optimizer, st_fn_front<Optimizer, Callable, BoundArgs...>, Callable,
          BoundArgs...> {
 public:
  using parent_type =
      self_tuning_function<Optimizer,
                           st_fn_front<Optimizer, Callable, BoundArgs...>,
                           Callable, BoundArgs...>;
  using parent_type::parent_type;

  friend parent_type;

 private:
  template <typename... FreeArgs, size_t... Seq>
  constexpr decltype(auto)
  call(std::index_sequence<Seq...>, FreeArgs &&...args) noexcept(
      noexcept(parent_type::func(
          detail::sub_placeholder_by_value<tunable_param>{}
              [std::get<std::integral_constant<size_t, Seq>::value>(
                  parent_type::arguments)]...,
          std::forward<FreeArgs>(args)...))) {
    return parent_type::func(
        detail::sub_placeholder_by_value<tunable_param>{}
            [std::get<std::integral_constant<size_t, Seq>::value>(
                parent_type::arguments)]...,
        std::forward<FreeArgs>(args)...);
  }
};

/**
 * @brief std::bind_back like interface for self_tuning_function
 */
template <typename Optimizer, typename Callable, typename... BoundArgs>
class st_fn_back
    : public self_tuning_function<Optimizer,
                                  st_fn_back<Optimizer, Callable, BoundArgs...>,
                                  Callable, BoundArgs...> {
 public:
  using parent_type =
      self_tuning_function<Optimizer,
                           st_fn_back<Optimizer, Callable, BoundArgs...>,
                           Callable, BoundArgs...>;
  using parent_type::parent_type;

  friend parent_type;

 private:
  template <typename... FreeArgs, size_t... Seq>
  constexpr decltype(auto)
  call(std::index_sequence<Seq...>, FreeArgs &&...args) noexcept(
      noexcept(parent_type::func(
          std::forward<FreeArgs>(args)...,
          detail::sub_placeholder_by_value<tunable_param>{}
              [std::get<std::integral_constant<size_t, Seq>::value>(
                  parent_type::arguments)]...))) {
    return parent_type::func(
        std::forward<FreeArgs>(args)...,
        detail::sub_placeholder_by_value<tunable_param>{}
            [std::get<std::integral_constant<size_t, Seq>::value>(
                parent_type::arguments)]...);
  }
};

/**
 * @brief Factory function to provide a clean interface for creating a new
 * self tuning function wrapper.
 */
template <typename Optimizer = IncrementalSubGradient, typename Callable,
          typename... Args>
[[nodiscard]] constexpr auto self_tuning(Callable &&callable,
                                         Args &&...args) noexcept {
  return st_fn_placeholders<
      decltype(Optimizer::make(std::forward<Args>(args)...)), Callable,
      Args...>{std::forward<Callable>(callable), std::forward<Args>(args)...};
}

template <typename Optimizer = IncrementalSubGradient, typename Callable,
          typename... Args>
[[nodiscard]] constexpr auto st_front(Callable &&callable, Args &&...args) {
  return st_fn_front<decltype(Optimizer::make(std::forward<Args>(args)...)),
                     Callable, Args...>{std::forward<Callable>(callable),
                                        std::forward<Args>(args)...};
}

template <typename Optimizer = IncrementalSubGradient, typename Callable,
          typename... Args>
[[nodiscard]] constexpr auto st_back(Callable &&callable, Args &&...args) {
  return st_fn_back<decltype(Optimizer::make(std::forward<Args>(args)...)),
                    Callable, Args...>{std::forward<Callable>(callable),
                                       std::forward<Args>(args)...};
}

}  // namespace treble

#endif  // TREBLE_SELF_TUNING_FUNCTION_H