#include "astar.hpp"

#include <gtest/gtest.h>

#include <cmath>

#include "support.hpp"

namespace astrum::minsnap {
namespace {

using testing::make_course;
using testing::make_gate;

// One gate facing north, centred 2 m up at the origin.
Course one_gate() { return make_course({make_gate({0.0, 0.0, 2.0}, 0.0)}); }

TEST(FrameDistance, MatchesTheHardware) {
  const Course c = one_gate();
  const double h = c.opening_m / 2.0;
  // Inside the opening, equidistant from both uprights: the inner face is h away.
  EXPECT_NEAR(frame_distance(c, 0, {0.0, 0.0, 2.0}), h - 0.0, 1e-12);
  // 10 cm in front of the left upright's outer face.
  EXPECT_NEAR(frame_distance(c, 0, {-(h + c.bar_m + 0.1), 0.0, 2.0}), 0.1, 1e-12);
  // Above the top bar.
  EXPECT_NEAR(frame_distance(c, 0, {0.0, 0.0, 2.0 + h + c.bar_m + 0.25}), 0.25, 1e-12);
  // The sill is open: the lowest hardware is the uprights' feet at the sill height, so a
  // point straight below the centre is the sill-to-upright distance away, not blocked.
  EXPECT_GT(frame_distance(c, 0, {0.0, 0.0, 2.0 - 2.0}), h);
  // Inside the stock.
  EXPECT_EQ(frame_distance(c, 0, {-(h + c.bar_m / 2.0), 0.0, 2.0}), 0.0);
  EXPECT_TRUE(std::isinf(nearest_frame_distance(c, {0, 0, 0}, 0)));
}

TEST(AStar, ClearStraightLineIsReturnedAsIs) {
  const Course c = one_gate();
  const Hazards h{&c, 0.353, 0.3};
  const Vec3 a{0.0, -3.0, 2.0}, b{0.0, 3.0, 2.0};  // through the middle of the opening
  const auto path = astar_path(a, b, h, AStarParams{});
  ASSERT_TRUE(path);
  EXPECT_EQ(path->size(), 2u);
}

TEST(AStar, DetoursAroundAFrameThatBlocksTheLine) {
  const Course c = one_gate();
  const Hazards h{&c, 0.353, 0.3};
  // Straight through the left upright.
  const double x = -(c.opening_m / 2.0 + c.bar_m / 2.0);
  const Vec3 a{x, -3.0, 2.0}, b{x, 3.0, 2.0};
  ASSERT_FALSE(segment_clear(a, b, h));
  const auto path = astar_path(a, b, h, AStarParams{});
  ASSERT_TRUE(path);
  EXPECT_GT(path->size(), 2u);
  EXPECT_EQ(path->front(), a);
  EXPECT_EQ(path->back(), b);
  double length = 0.0;
  for (std::size_t i = 0; i + 1 < path->size(); ++i) {
    EXPECT_TRUE(segment_clear((*path)[i], (*path)[i + 1], h)) << "segment " << i;
    length += norm((*path)[i + 1] - (*path)[i]);
  }
  EXPECT_GT(length, 6.0);
  EXPECT_LT(length, 6.0 + 2.5);  // a short detour, not a lap round the course
}

TEST(AStar, KeepsAboveTheFloor) {
  const Course c = make_course({});
  const Hazards h{&c, 0.353, 1.0};
  const Vec3 a{0.0, 0.0, 1.2}, b{6.0, 0.0, 1.2};
  const auto path = astar_path(a, b, h, AStarParams{});
  ASSERT_TRUE(path);
  EXPECT_TRUE(segment_clear(a, b, h));
  EXPECT_EQ(path->size(), 2u);
  EXPECT_FALSE(segment_clear({0, 0, 0.5}, {1, 0, 0.5}, h));
}

TEST(AStar, ReportsNoRouteWhenTheBoxIsBlocked) {
  const Course c = make_course({});
  const Hazards h{&c, 0.353, 10.0};  // every cell is below the floor
  EXPECT_FALSE(astar_path({0, 0, 2}, {5, 0, 2}, h, AStarParams{}));
}

}  // namespace
}  // namespace astrum::minsnap
