# Treble
A header-only library for self-tuning functions.

This was originally developed for finding optimal cuda kernel launch parameters, but can be used
anywhere you want find optimal parameters for functions on your hot-path.

```cpp
// Our function to optimize
auto test_func = [](double x, int ms) noexcept -> double {
    std::cout << "Sleeping for " << 10 * ms << " milliseconds" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds{10 * ms});
    return x * x;
};

// Create our self-tuning function wrapper.
// Every time this function is called it runs a single iteration of the optimization routine
// Forwards the given arguments and current trial values of the paramters to the wrapped function and propagates the return value (if any).
auto func = treble::self_tuning(test_func, std::placeholders::_1,
                                treble::tunable_param{30/*starting value*/, 0/*min*/, 50/*max*/, 5/*step size*/});

// Counterparts to std::bind_front and std::bind_back are also available
auto func2 = treble::self_tuning_back(test_func, treble::tunable_param{30, 0, 50, 5});
```

Provides a std::bind like interface for creating self-tuning functions with integer parameters.

Currently the only supported objective function is minimizing the execution time of the wrapped function.