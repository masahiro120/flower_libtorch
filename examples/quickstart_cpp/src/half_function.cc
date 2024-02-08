#include "../../../tiny-dnn/tiny_dnn/tiny_dnn.h"

std::vector<half> one_vector_to_half(const tiny_dnn::vec_t& array) {
    std::vector<half> array_half(array.size());
    // for (size_t i = 0; i < array.size(); ++i) {
    tiny_dnn::for_i(array.size(), [&](size_t i) {
        array_half[i] = half(array[i]);
    });
    // }

    return array_half;
}

std::vector<half> one_vector_to_half(const std::vector<size_t>& array) {
    std::vector<half> array_half(array.size());
    // for (size_t i = 0; i < array.size(); ++i) {
    tiny_dnn::for_i(array.size(), [&](size_t i) {
        array_half[i] = half(array[i]);
    });
    // }

    return array_half;
}

std::vector<std::vector<half>> two_vector_to_half(const tiny_dnn::tensor_t& array) {
    std::vector<std::vector<half>> array_half(array.size());
    // for (size_t i = 0; i < array.size(); ++i) {
    tiny_dnn::for_i(array.size(), [&](size_t i) {
        array_half[i].resize(array[i].size());

        for (size_t j = 0; j < array[i].size(); ++j) {
            array_half[i][j] = half(array[i][j]);
        }
    });
    // }

    return array_half;
}

std::vector<std::vector<half>> two_vector_to_half(const std::vector<std::vector<size_t>>& array) {
    std::vector<std::vector<half>> array_half(array.size());
    // for (size_t i = 0; i < array.size(); ++i) {
    tiny_dnn::for_i(array.size(), [&](size_t i) {
        array_half[i].resize(array[i].size());

        for (size_t j = 0; j < array[i].size(); ++j) {
            array_half[i][j] = half(array[i][j]);
        }
    });
    // }

    return array_half;
}

std::vector<std::vector<std::vector<half>>> three_vector_to_half(const std::vector<tiny_dnn::tensor_t>& array) {
    std::vector<std::vector<std::vector<half>>> array_half(array.size());
    // for (size_t i = 0; i < array.size(); ++i) {
    tiny_dnn::for_i(array.size(), [&](size_t i) {
        array_half[i].resize(array[i].size());

        for (size_t j = 0; j < array[i].size(); ++j) {
            array_half[i][j].resize(array[i][j].size());

            for (size_t k = 0; k < array[i][j].size(); ++k) {
                array_half[i][j][k] = half(array[i][j][k]);
            }
        }
    });
    // }

    return array_half;
}


void one_half_to_vector(tiny_dnn::vec_t& array, std::vector<half> array_half) {
  // for (size_t i = 0; i < array.size(); ++i) {
  tiny_dnn::for_i(array.size(), [&](size_t i) {
    array[i] = static_cast<float>(array_half[i]);
  });
  // }
}

void two_half_to_vector(tiny_dnn::tensor_t& array, std::vector<std::vector<half>> array_half) {
  // for (size_t i = 0; i < array.size(); ++i) {
  tiny_dnn::for_i(array.size(), [&](size_t i) {
    for (size_t j = 0; j < array[i].size(); ++j) {
      array[i][j] = static_cast<float>(array_half[i][j]);
    }
  });
  // }
}

void two_half_to_vector(std::vector<std::vector<size_t>>& array, std::vector<std::vector<half>> array_half) {
  // for (size_t i = 0; i < array.size(); ++i) {
  tiny_dnn::for_i(array.size(), [&](size_t i) {
    for (size_t j = 0; j < array[i].size(); ++j) {
      array[i][j] = static_cast<float>(array_half[i][j]);
    }
  });
  // }
}

void three_half_to_vector(std::vector<tiny_dnn::tensor_t>& array, std::vector<std::vector<std::vector<half>>> array_half) {
  // for (size_t i = 0; i < array.size(); ++i) {
  tiny_dnn::for_i(array.size(), [&](size_t i) {
    for (size_t j = 0; j < array[i].size(); ++j) {
      for (size_t k = 0; k < array[i][j].size(); ++k) {
        array[i][j][k] = static_cast<float>(array_half[i][j][k]);
      }
    }
  });
  // }
}

void vector_div_half(std::vector<half> &x, half denom) {
  std::transform(x.begin(), x.end(), x.begin(),
                 [=](half x) { return x / denom; });
}

void moments_impl_calc_mean_half(size_t num_examples,
                            size_t channels,
                            size_t spatial_dim,
                            const std::vector<std::vector<half>> &in,
                            std::vector<half> &mean) {
  for (size_t i = 0; i < num_examples; i++) {
    for (size_t j = 0; j < channels; j++) {
      half &rmean = mean.at(j);
      const auto it  = in[i].begin() + (j * spatial_dim);
      rmean          = std::accumulate(it, it + spatial_dim, rmean);
    }
  }
}

void moments_impl_calc_variance(size_t num_examples,
                                size_t channels,
                                size_t spatial_dim,
                                const std::vector<std::vector<half>> &in,
                                const std::vector<half> &mean,
                                std::vector<half> &variance) {
  assert(mean.size() >= channels);
  for (size_t i = 0; i < num_examples; i++) {
    for (size_t j = 0; j < channels; j++) {
      half &rvar    = variance[j];
      const auto it    = in[i].begin() + (j * spatial_dim);
      const half ex = mean[j];
      rvar             = std::accumulate(it, it + spatial_dim, rvar,
                             [ex](half current, half x) {
                               return current + pow(x - ex, half{2.0});
                             });
    }
  }
  // vector_div_half(
  //   variance,
  //   std::max(half{1.0f}, half(num_examples * spatial_dim) - half{1.0f}));

  if (half(num_examples * spatial_dim) - half{1.0f} < half{1.0f}) {
    vector_div_half(variance, half{1.0f});
  } else {
    vector_div_half(variance, half(num_examples * spatial_dim) - half{1.0f});
  }
}

void moments_half(const std::vector<std::vector<half>> &in,
                    size_t spatial_dim,
                    size_t channels,
                    std::vector<half> &mean) {
  const size_t num_examples = in.size();
  assert(in[0].size() == spatial_dim * channels);

  mean.resize(channels);
  // vectorize::fill(&mean[0], mean.size(), float_t{0.0});
  for (size_t i = 0; i < mean.size(); ++i) {
    mean[i] = half(0.0);
  }
  moments_impl_calc_mean_half(num_examples, channels, spatial_dim, in, mean);
  vector_div_half(mean, half(num_examples * spatial_dim));
}

void moments_half(const std::vector<std::vector<half>> &in,
                    size_t spatial_dim,
                    size_t channels,
                    std::vector<half> &mean,
                    std::vector<half> &variance) {
  const size_t num_examples = in.size();
  assert(in[0].size() == spatial_dim * channels);

  // calc mean
  moments_half(in, spatial_dim, channels, mean);

  variance.resize(channels);
  // vectorize::fill(&variance[0], variance.size(), float_t{0.0});
  for (size_t i = 0; i < variance.size(); ++i) {
    variance[i] = half(0.0);
  }
  moments_impl_calc_variance(num_examples, channels, spatial_dim, in,
                                     mean, variance);
}

class random_generator {
 public:
  static random_generator &get_instance() {
    static random_generator instance;
    return instance;
  }

  std::mt19937 &operator()() { return gen_; }

  void set_seed(unsigned int seed) { gen_.seed(seed); }

 private:
  // avoid gen_(0) for MSVC known issue
  // https://connect.microsoft.com/VisualStudio/feedback/details/776456
  random_generator() : gen_(1) {}
  std::mt19937 gen_;
};

half uniform_rand_half(half min, half max) {
  // std::uniform_real_distribution<half> dst(min, max);
  std::uniform_real_distribution<double> dst(static_cast<double>(min), static_cast<double>(max));
  return half(dst(random_generator::get_instance()()));
}


bool bernoulli_half(half p) {
  return uniform_rand_half(half{0}, half{1}) <= p;
}

void apply_cost_if_defined_half(std::vector<std::vector<half>> &sample_gradient, const std::vector<std::vector<half>> &sample_cost) {
  if (sample_gradient.size() == sample_cost.size()) {
    // @todo consider adding parallelism
    const size_t channel_count = sample_gradient.size();
    for (size_t channel = 0; channel < channel_count; ++channel) {
      if (sample_gradient[channel].size() == sample_cost[channel].size()) {
        const size_t element_count = sample_gradient[channel].size();

        // @todo optimize? (use AVX or so)
        for (size_t element = 0; element < element_count; ++element) {
          sample_gradient[channel][element] *= sample_cost[channel][element];
        }
      }
    }
  }
}

// cross-entropy loss function for multi-class classification
class cross_entropy_multiclass_half {
 public:
  static half f(const std::vector<half> &y, const std::vector<half> &t) {
    assert(y.size() == t.size());
    half d{0.0};
    half epsilon = half(1e-6); // small positive value to avoid log zero
    for (size_t i = 0; i < y.size(); ++i) d += -t[i] * std::log(y[i]+epsilon);

    return d;
  }

  static std::vector<half> df(const std::vector<half> &y, const std::vector<half> &t) {
    assert(y.size() == t.size());
    half epsilon = half(1e-6); // small positive value to avoid log zero
    std::vector<half> d(t.size());

    for (size_t i = 0; i < y.size(); ++i) d[i] = -t[i] / (y[i]+epsilon);

    return d;
  }
};


std::vector<half> gradient_half(const std::vector<half> &y, const std::vector<half> &t) {
  assert(y.size() == t.size());
  return cross_entropy_multiclass_half::df(y, t);
}

std::vector<std::vector<half>> gradient_half(const std::vector<std::vector<half>> &y,
                                             const std::vector<std::vector<half>> &t) {
  std::vector<std::vector<half>> grads(y.size());

  assert(y.size() == t.size());

  // printf("gradient_half::: 1\n");

  for (size_t i = 0; i < y.size(); i++) grads[i] = gradient_half(y[i], t[i]);

  // printf("gradient_half::: 2\n");
  return grads;
}

std::vector<std::vector<std::vector<half>>> gradient_half(
                               const std::vector<std::vector<std::vector<half>>> &y,
                               const std::vector<std::vector<std::vector<half>>> &t,
                               const std::vector<std::vector<std::vector<half>>> &t_cost) {
  // printf("gradient_half\n");
  const size_t sample_count  = y.size();
  const size_t channel_count = y[0].size();

  std::vector<std::vector<std::vector<half>>> gradients(sample_count);

  CNN_UNREFERENCED_PARAMETER(channel_count);
  assert(y.size() == t.size());
  assert(t_cost.empty() || t_cost.size() == t.size());

  // printf("gradient_half 2\n");
  // printf("sample_count: %d\n", sample_count);
  // @todo add parallelism
  for (size_t sample = 0; sample < sample_count; ++sample) {
    assert(y[sample].size() == channel_count);
    assert(t[sample].size() == channel_count);
    assert(t_cost.empty() || t_cost[sample].empty() ||
           t_cost[sample].size() == channel_count);

    gradients[sample] = gradient_half(y[sample], t[sample]);

    if (sample < t_cost.size()) {
      apply_cost_if_defined_half(gradients[sample], t_cost[sample]);
    }
  }
  // printf("gradient_half:::: 2\n");
  // printf("gradient size: %d\n", gradients.size());


  return gradients;
}