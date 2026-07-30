/*
 * Copyright 2015 Yu-Chin Juan, Wei-Sheng Chin, and Yong Zhuang.
 *
 * Licensed under the Apache License, Version 2.0. See LICENSE in this
 * directory. This implementation retains the field-aware online optimizer
 * and extends its sparse reader for weighted profile and causal-history
 * fields.
 */

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>
#include <pmmintrin.h>

namespace {

constexpr int kMaxLineSize = 1000000;
constexpr std::uint32_t kBaseFieldCount = 15;
constexpr std::uint32_t kMaximumToken = 999999;
constexpr std::uint32_t kBasisPointDenominator = 10000;
constexpr std::uint32_t kWeightNodeSize = 2;
constexpr float kBaseWeight = 0.36514837167011077179F;

struct DataNode {
    DataNode(
        std::uint32_t const field,
        std::uint32_t const feature,
        float const value
    ) : field(field), feature(feature), value(value) {}

    std::uint32_t field;
    std::uint32_t feature;
    float value;
};

struct Problem {
    std::uint32_t features = 0;
    std::uint32_t fields = 0;
    std::vector<DataNode> nodes;
    std::vector<std::uint64_t> offsets;
    std::vector<float> labels;
};

struct ReadOptions {
    std::uint32_t publisher_mask_basis_points = 0;
    std::uint32_t cold_publisher_token = 1;
    bool force_cold_publisher = false;
};

struct Options {
    std::string train_path;
    std::string score_path;
    std::string output_path;
    float learning_rate = 0.05F;
    float l2 = 0.00002F;
    std::uint32_t rank = 4;
    std::uint32_t epochs = 6;
    std::uint32_t publisher_mask_basis_points = 0;
    std::uint32_t cold_publisher_token = 1;
    bool score_cold_publisher = false;
};

struct Model {
    Model(
        std::uint32_t const features,
        std::uint32_t const rank,
        std::uint32_t const fields
    ) : weights(
            static_cast<std::uint64_t>(features)
                * fields
                * rank
                * kWeightNodeSize,
            0.0F
        ),
        features(features),
        rank(rank),
        fields(fields) {}

    std::vector<float> weights;
    std::uint32_t const features;
    std::uint32_t const rank;
    std::uint32_t const fields;
};

std::string help() {
    return
        "usage: profile-ffm-solver --train PATH --score PATH --output PATH "
        "[options]\n"
        "\n"
        "options:\n"
        "  --learning-rate FLOAT\n"
        "  --l2 FLOAT\n"
        "  --rank INTEGER\n"
        "  --epochs INTEGER\n"
        "  --publisher-mask-bp INTEGER\n"
        "  --cold-publisher-token INTEGER\n"
        "  --score-cold-publisher\n";
}

std::string require_value(
    std::vector<std::string> const &arguments,
    std::size_t &index
) {
    if(index + 1 >= arguments.size()) {
        throw std::invalid_argument("missing option value\n");
    }
    return arguments[++index];
}

Options parse_options(int const argc, char const * const * const argv) {
    Options options;
    std::vector<std::string> arguments;
    for(int index = 1; index < argc; ++index) {
        arguments.emplace_back(argv[index]);
    }
    for(std::size_t index = 0; index < arguments.size(); ++index) {
        std::string const &argument = arguments[index];
        if(argument == "--train") {
            options.train_path = require_value(arguments, index);
        } else if(argument == "--score") {
            options.score_path = require_value(arguments, index);
        } else if(argument == "--output") {
            options.output_path = require_value(arguments, index);
        } else if(argument == "--learning-rate") {
            options.learning_rate = std::stof(require_value(arguments, index));
        } else if(argument == "--l2") {
            options.l2 = std::stof(require_value(arguments, index));
        } else if(argument == "--rank") {
            options.rank = static_cast<std::uint32_t>(
                std::stoul(require_value(arguments, index))
            );
        } else if(argument == "--epochs") {
            options.epochs = static_cast<std::uint32_t>(
                std::stoul(require_value(arguments, index))
            );
        } else if(argument == "--publisher-mask-bp") {
            options.publisher_mask_basis_points = static_cast<std::uint32_t>(
                std::stoul(require_value(arguments, index))
            );
        } else if(argument == "--cold-publisher-token") {
            options.cold_publisher_token = static_cast<std::uint32_t>(
                std::stoul(require_value(arguments, index))
            );
        } else if(argument == "--score-cold-publisher") {
            options.score_cold_publisher = true;
        } else {
            throw std::invalid_argument("unknown option: " + argument + "\n");
        }
    }
    if(
        options.train_path.empty()
        || options.score_path.empty()
        || options.output_path.empty()
    ) {
        throw std::invalid_argument(help());
    }
    if(
        options.rank == 0
        || options.rank % 4 != 0
        || options.epochs == 0
        || !std::isfinite(options.learning_rate)
        || options.learning_rate <= 0.0F
        || !std::isfinite(options.l2)
        || options.l2 < 0.0F
        || options.publisher_mask_basis_points > kBasisPointDenominator
        || options.cold_publisher_token == 0
        || options.cold_publisher_token > kMaximumToken
    ) {
        throw std::invalid_argument("invalid solver option\n");
    }
    return options;
}

std::FILE *open_file(std::string const &path, char const *mode) {
    std::FILE *file = std::fopen(path.c_str(), mode);
    if(file == nullptr) {
        throw std::runtime_error("cannot open " + path);
    }
    return file;
}

bool deterministic_mask(
    std::uint32_t const row,
    std::uint32_t const basis_points
) {
    if(basis_points == 0) {
        return false;
    }
    std::uint64_t value =
        static_cast<std::uint64_t>(row) + 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    value ^= value >> 31;
    return value % kBasisPointDenominator < basis_points;
}

std::uint32_t parse_compact_token(char const *text) {
    errno = 0;
    char *end = nullptr;
    unsigned long const value = std::strtoul(text, &end, 10);
    if(
        errno != 0
        || end == text
        || *end != '\0'
        || value == 0
        || value > kMaximumToken
    ) {
        throw std::runtime_error("invalid compact token");
    }
    return static_cast<std::uint32_t>(value);
}

void parse_weighted_token(
    char const *text,
    std::uint32_t &field,
    std::uint32_t &feature,
    float &value
) {
    char trailing;
    if(
        std::sscanf(text, "%u:%u:%f%c", &field, &feature, &value, &trailing)
        != 3
    ) {
        throw std::runtime_error("invalid weighted token");
    }
    if(field < 16 || field > 19) {
        throw std::runtime_error("weighted fields must be in [16, 19]");
    }
    if(
        feature == 0
        || feature > kMaximumToken
        || !std::isfinite(value)
        || value <= 0.0F
    ) {
        throw std::runtime_error("invalid weighted feature value");
    }
}

Problem read_problem(std::string const &path, ReadOptions const &options) {
    Problem problem;
    std::FILE *file = open_file(path, "r");
    char line[kMaxLineSize];
    std::uint64_t offset = 0;
    problem.offsets.push_back(0);

    for(
        std::uint32_t row = 0;
        std::fgets(line, kMaxLineSize, file) != nullptr;
        ++row
    ) {
        std::size_t const length = std::strlen(line);
        if(
            length == static_cast<std::size_t>(kMaxLineSize - 1)
            && line[length - 1] != '\n'
            && !std::feof(file)
        ) {
            throw std::runtime_error("sparse input line is too long");
        }
        char *label = std::strtok(line, " \t\r\n");
        if(
            label == nullptr
            || !(
                std::strcmp(label, "0") == 0
                || std::strcmp(label, "1") == 0
            )
        ) {
            throw std::runtime_error("label must be 0 or 1");
        }
        problem.labels.push_back(label[0] == '1' ? 1.0F : -1.0F);

        bool const mask_publisher =
            options.force_cold_publisher
            || deterministic_mask(row, options.publisher_mask_basis_points);
        std::uint32_t compact_field = 1;
        bool weighted_started = false;
        for(
            char *token = std::strtok(nullptr, " \t\r\n");
            token != nullptr;
            token = std::strtok(nullptr, " \t\r\n")
        ) {
            std::uint32_t field;
            std::uint32_t feature;
            float value;
            if(std::strchr(token, ':') == nullptr) {
                if(weighted_started) {
                    throw std::runtime_error(
                        "compact token follows a weighted token"
                    );
                }
                if(compact_field > kBaseFieldCount) {
                    throw std::runtime_error("too many compact fields");
                }
                field = compact_field++;
                feature = parse_compact_token(token);
                if(field == 1 && mask_publisher) {
                    feature = options.cold_publisher_token;
                }
                value = kBaseWeight;
            } else {
                weighted_started = true;
                if(compact_field != kBaseFieldCount + 1) {
                    throw std::runtime_error(
                        "weighted token precedes the 15 compact fields"
                    );
                }
                parse_weighted_token(token, field, feature, value);
            }
            problem.fields = std::max(problem.fields, field);
            problem.features = std::max(problem.features, feature);
            problem.nodes.emplace_back(field - 1, feature - 1, value);
            ++offset;
        }
        if(compact_field != kBaseFieldCount + 1) {
            throw std::runtime_error(
                "sparse row does not contain exactly 15 compact fields"
            );
        }
        problem.offsets.push_back(offset);
    }
    std::fclose(file);
    if(problem.labels.empty()) {
        throw std::runtime_error("sparse input is empty");
    }
    return problem;
}

float score_instance(
    Problem const &problem,
    Model &model,
    std::uint32_t const row,
    float const kappa = 0.0F,
    float const learning_rate = 0.0F,
    float const l2 = 0.0F,
    bool const update = false
) {
    std::uint32_t const rank = model.rank;
    std::uint64_t const factor_stride = rank * kWeightNodeSize;
    std::uint64_t const feature_stride = model.fields * factor_stride;
    float * const weights = model.weights.data();

    __m128 const vector_kappa = _mm_set1_ps(kappa);
    __m128 const vector_learning_rate = _mm_set1_ps(learning_rate);
    __m128 const vector_l2 = _mm_set1_ps(l2);
    __m128 total = _mm_setzero_ps();

    for(
        std::uint64_t left = problem.offsets[row];
        left < problem.offsets[row + 1];
        ++left
    ) {
        DataNode const &left_node = problem.nodes[left];
        if(
            left_node.feature >= model.features
            || left_node.field >= model.fields
        ) {
            continue;
        }
        for(
            std::uint64_t right = left + 1;
            right < problem.offsets[row + 1];
            ++right
        ) {
            DataNode const &right_node = problem.nodes[right];
            if(
                right_node.feature >= model.features
                || right_node.field >= model.fields
                || left_node.field == right_node.field
            ) {
                continue;
            }
            float * const left_weights =
                weights
                + left_node.feature * feature_stride
                + right_node.field * factor_stride;
            float * const right_weights =
                weights
                + right_node.feature * feature_stride
                + left_node.field * factor_stride;
            __m128 const value = _mm_set1_ps(
                left_node.value * right_node.value
            );

            if(update) {
                __m128 const weighted_kappa = _mm_mul_ps(vector_kappa, value);
                float * const left_accumulator = left_weights + rank;
                float * const right_accumulator = right_weights + rank;
                for(std::uint32_t factor = 0; factor < rank; factor += 4) {
                    __m128 left_factor = _mm_load_ps(left_weights + factor);
                    __m128 right_factor = _mm_load_ps(right_weights + factor);
                    __m128 left_sum = _mm_load_ps(left_accumulator + factor);
                    __m128 right_sum = _mm_load_ps(right_accumulator + factor);
                    __m128 const left_gradient = _mm_add_ps(
                        _mm_mul_ps(vector_l2, left_factor),
                        _mm_mul_ps(weighted_kappa, right_factor)
                    );
                    __m128 const right_gradient = _mm_add_ps(
                        _mm_mul_ps(vector_l2, right_factor),
                        _mm_mul_ps(weighted_kappa, left_factor)
                    );
                    left_sum = _mm_add_ps(
                        left_sum,
                        _mm_mul_ps(left_gradient, left_gradient)
                    );
                    right_sum = _mm_add_ps(
                        right_sum,
                        _mm_mul_ps(right_gradient, right_gradient)
                    );
                    left_factor = _mm_sub_ps(
                        left_factor,
                        _mm_mul_ps(
                            vector_learning_rate,
                            _mm_mul_ps(_mm_rsqrt_ps(left_sum), left_gradient)
                        )
                    );
                    right_factor = _mm_sub_ps(
                        right_factor,
                        _mm_mul_ps(
                            vector_learning_rate,
                            _mm_mul_ps(_mm_rsqrt_ps(right_sum), right_gradient)
                        )
                    );
                    _mm_store_ps(left_weights + factor, left_factor);
                    _mm_store_ps(right_weights + factor, right_factor);
                    _mm_store_ps(left_accumulator + factor, left_sum);
                    _mm_store_ps(right_accumulator + factor, right_sum);
                }
            } else {
                for(std::uint32_t factor = 0; factor < rank; factor += 4) {
                    total = _mm_add_ps(
                        total,
                        _mm_mul_ps(
                            _mm_mul_ps(
                                _mm_load_ps(left_weights + factor),
                                _mm_load_ps(right_weights + factor)
                            ),
                            value
                        )
                    );
                }
            }
        }
    }
    if(update) {
        return 0.0F;
    }
    total = _mm_hadd_ps(total, total);
    total = _mm_hadd_ps(total, total);
    float result;
    _mm_store_ss(&result, total);
    return result;
}

void initialize(Model &model) {
    float const coefficient =
        static_cast<float>(0.5 / std::sqrt(static_cast<double>(model.rank)));
    float *weight = model.weights.data();
    for(std::uint32_t feature = 0; feature < model.features; ++feature) {
        for(std::uint32_t field = 0; field < model.fields; ++field) {
            for(
                std::uint32_t factor = 0;
                factor < model.rank;
                ++factor, ++weight
            ) {
                *weight = coefficient * static_cast<float>(drand48());
            }
            for(
                std::uint32_t factor = model.rank;
                factor < 2 * model.rank;
                ++factor, ++weight
            ) {
                *weight = 1.0F;
            }
        }
    }
}

float predict(
    Problem const &problem,
    Model &model,
    std::string const &output_path
) {
    std::FILE *output = nullptr;
    if(!output_path.empty()) {
        output = open_file(output_path, "w");
    }
    double loss = 0.0;
    #pragma omp parallel for schedule(static) reduction(+:loss)
    for(std::uint32_t row = 0; row < problem.labels.size(); ++row) {
        float const label = problem.labels[row];
        float const score = score_instance(problem, model, row);
        float const probability =
            1.0F / (1.0F + static_cast<float>(std::exp(-score)));
        float const exponential = static_cast<float>(
            std::exp(-label * score)
        );
        loss += std::log(1.0F + exponential);
        if(output != nullptr) {
            std::fprintf(output, "%lf\n", probability);
        }
    }
    if(output != nullptr) {
        std::fclose(output);
    }
    return static_cast<float>(
        loss / static_cast<double>(problem.labels.size())
    );
}

void train(
    Problem const &training,
    Model &model,
    Options const &options
) {
    std::cout << "epoch train_logloss\n";
    for(std::uint32_t epoch = 0; epoch < options.epochs; ++epoch) {
        double training_loss = 0.0;
        #pragma omp parallel for schedule(static) reduction(+:training_loss)
        for(std::uint32_t row = 0; row < training.labels.size(); ++row) {
            float const label = training.labels[row];
            float const score = score_instance(training, model, row);
            float const exponential = static_cast<float>(
                std::exp(-label * score)
            );
            training_loss += std::log(1.0F + exponential);
            float const kappa =
                -label * exponential / (1.0F + exponential);
            score_instance(
                training,
                model,
                row,
                kappa,
                options.learning_rate,
                options.l2,
                true
            );
        }
        training_loss /= static_cast<double>(training.labels.size());
        std::cout
            << epoch
            << ' '
            << training_loss
            << '\n'
            << std::flush;
    }
}

} // namespace

int main(int const argc, char const * const * const argv) {
    try {
        Options const options = parse_options(argc, argv);
        omp_set_num_threads(1);
        Problem const scoring = read_problem(
            options.score_path,
            ReadOptions{
                0,
                options.cold_publisher_token,
                options.score_cold_publisher,
            }
        );
        Problem const training = read_problem(
            options.train_path,
            ReadOptions{
                options.publisher_mask_basis_points,
                options.cold_publisher_token,
                false,
            }
        );
        Model model(training.features, options.rank, training.fields);
        initialize(model);
        train(training, model, options);
        predict(scoring, model, options.output_path);
    } catch(std::exception const &error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
