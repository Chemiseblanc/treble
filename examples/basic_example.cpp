#include <cstdlib>
#include <iostream>
#include <thread>

#include "treble/treble.hpp"

int main(int, char **) {
  auto test_func = [](double x, int ms) noexcept -> double {
    std::cout << "Sleeping for " << 10 * ms << " milliseconds" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds{10 * ms});
    return x * x;
  };

  auto func = treble::self_tuning(test_func, std::placeholders::_1,
                                  treble::tunable_param{30, 0, 50, 5});

  for (int i = 0; i < 30; ++i) {
    std::cout << func(static_cast<double>(i)) << std::endl;
  }
  return EXIT_SUCCESS;
}