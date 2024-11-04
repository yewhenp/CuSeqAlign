#pragma once

#include <Eigen/Dense>
#include <tuple>
#include <vector>

using ScoreType = int;

using ScoreMatrix = Eigen::Matrix<ScoreType, -1, -1, Eigen::ColMajor>;
using TraceMatrix = Eigen::Matrix<char, -1, -1, Eigen::ColMajor>;
