/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/memory_space_assignment/utils.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace xla {
namespace memory_space_assignment {
namespace {

using ::testing::ElementsAreArray;

class BreadthFirstMidpointIteratorTest : public ::testing::Test {
 protected:
  static void RunTest(int start, int end, std::vector<int> expected_order) {
    std::vector<int> actual;
    for (BreadthFirstMidpointIterator iterator(start, end); !iterator.End();
         iterator.Next()) {
      actual.push_back(iterator.value());
    }
    EXPECT_THAT(actual, ElementsAreArray(expected_order));
  }
};

TEST_F(BreadthFirstMidpointIteratorTest, NoValues) { RunTest(1, 0, {}); }

TEST_F(BreadthFirstMidpointIteratorTest, OneValue) { RunTest(1, 1, {1}); }

TEST_F(BreadthFirstMidpointIteratorTest, TwoValues) { RunTest(1, 2, {2, 1}); }

TEST_F(BreadthFirstMidpointIteratorTest, General1) {
  RunTest(1, 5, {3, 2, 5, 1, 4});
}

TEST_F(BreadthFirstMidpointIteratorTest, General2) {
  RunTest(0, 10, {5, 2, 8, 1, 4, 7, 10, 0, 3, 6, 9});
}

}  // namespace
}  // namespace memory_space_assignment
}  // namespace xla
