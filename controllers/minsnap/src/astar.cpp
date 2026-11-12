#include "astar.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <queue>

namespace astrum::minsnap {

double Hazards::margin(const Vec3& p) const {
  double m = p.z - floor_m;
  if (course != nullptr) {
    m = std::min(m, nearest_frame_distance(*course, p, course->gates.size()) - hull_radius_m);
  }
  return m;
}

bool segment_clear(const Vec3& a, const Vec3& b, const Hazards& hazards, double step_m) {
  const double len = norm(b - a);
  const int steps = std::max(1, static_cast<int>(std::ceil(len / step_m)));
  for (int i = 0; i <= steps; ++i) {
    if (hazards.margin(a + (b - a) * (static_cast<double>(i) / steps)) < 0.0) return false;
  }
  return true;
}

namespace {

struct Grid {
  Vec3 origin;
  double cell = 0.0;
  std::array<int, 3> dims{};

  std::size_t index(const std::array<int, 3>& c) const {
    return (static_cast<std::size_t>(c[2]) * static_cast<std::size_t>(dims[1]) +
            static_cast<std::size_t>(c[1])) *
               static_cast<std::size_t>(dims[0]) +
           static_cast<std::size_t>(c[0]);
  }
  bool inside(const std::array<int, 3>& c) const {
    return c[0] >= 0 && c[1] >= 0 && c[2] >= 0 && c[0] < dims[0] && c[1] < dims[1] &&
           c[2] < dims[2];
  }
  Vec3 centre(const std::array<int, 3>& c) const {
    return origin + Vec3{(c[0] + 0.5) * cell, (c[1] + 0.5) * cell, (c[2] + 0.5) * cell};
  }
  std::array<int, 3> cell_of(const Vec3& p) const {
    const Vec3 r = (p - origin) / cell;
    auto clampi = [](double v, int hi) {
      return std::clamp(static_cast<int>(std::floor(v)), 0, hi - 1);
    };
    return {clampi(r.x, dims[0]), clampi(r.y, dims[1]), clampi(r.z, dims[2])};
  }
};

}  // namespace

std::optional<std::vector<Vec3>> astar_path(const Vec3& start, const Vec3& goal,
                                            const Hazards& hazards, const AStarParams& params) {
  if (segment_clear(start, goal, hazards)) return std::vector<Vec3>{start, goal};

  Grid grid;
  grid.cell = params.cell_m;
  const Vec3 lo{std::min(start.x, goal.x) - params.box_margin_m,
                std::min(start.y, goal.y) - params.box_margin_m,
                std::min(start.z, goal.z) - params.box_margin_m};
  const Vec3 hi{std::max(start.x, goal.x) + params.box_margin_m,
                std::max(start.y, goal.y) + params.box_margin_m,
                std::max(start.z, goal.z) + params.box_margin_m};
  grid.origin = lo;
  for (int a = 0; a < 3; ++a)
    grid.dims[static_cast<std::size_t>(a)] =
        static_cast<int>(std::ceil((hi[a] - lo[a]) / grid.cell));
  const std::size_t cells = static_cast<std::size_t>(grid.dims[0]) *
                            static_cast<std::size_t>(grid.dims[1]) *
                            static_cast<std::size_t>(grid.dims[2]);
  if (cells == 0 || cells > params.max_cells) return std::nullopt;

  const auto start_cell = grid.cell_of(start);
  const auto goal_cell = grid.cell_of(goal);
  const double block_below = 0.87 * grid.cell;  // half the cell diagonal

  // 0 unknown, 1 free, 2 blocked.
  std::vector<std::uint8_t> state(cells, 0);
  auto blocked = [&](const std::array<int, 3>& c) {
    std::uint8_t& s = state[grid.index(c)];
    if (s == 0) {
      const bool endpoint = c == start_cell || c == goal_cell;
      s = (endpoint || hazards.margin(grid.centre(c)) >= block_below) ? 1 : 2;
    }
    return s == 2;
  };

  std::vector<double> g(cells, std::numeric_limits<double>::infinity());
  std::vector<std::int64_t> parent(cells, -1);
  using Item = std::pair<double, std::size_t>;
  std::priority_queue<Item, std::vector<Item>, std::greater<>> open;
  auto h = [&](const std::array<int, 3>& c) {
    return norm(grid.centre(c) - grid.centre(goal_cell));
  };

  g[grid.index(start_cell)] = 0.0;
  open.emplace(h(start_cell), grid.index(start_cell));
  const std::size_t goal_index = grid.index(goal_cell);
  auto unindex = [&](std::size_t idx) {
    const std::size_t nx = static_cast<std::size_t>(grid.dims[0]);
    const std::size_t ny = static_cast<std::size_t>(grid.dims[1]);
    return std::array<int, 3>{static_cast<int>(idx % nx), static_cast<int>((idx / nx) % ny),
                              static_cast<int>(idx / (nx * ny))};
  };

  bool found = false;
  while (!open.empty()) {
    const auto [f, idx] = open.top();
    open.pop();
    const auto c = unindex(idx);
    if (f > g[idx] + h(c) + 1e-9) continue;  // stale entry
    if (idx == goal_index) {
      found = true;
      break;
    }
    for (int dz = -1; dz <= 1; ++dz) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          if (dx == 0 && dy == 0 && dz == 0) continue;
          const std::array<int, 3> n{c[0] + dx, c[1] + dy, c[2] + dz};
          if (!grid.inside(n) || blocked(n)) continue;
          const std::size_t ni = grid.index(n);
          const double step =
              grid.cell * std::sqrt(static_cast<double>(dx * dx + dy * dy + dz * dz));
          if (g[idx] + step < g[ni]) {
            g[ni] = g[idx] + step;
            parent[ni] = static_cast<std::int64_t>(idx);
            open.emplace(g[ni] + h(n), ni);
          }
        }
      }
    }
  }
  if (!found) return std::nullopt;

  std::vector<Vec3> raw;
  for (std::int64_t i = static_cast<std::int64_t>(goal_index); i >= 0;
       i = parent[static_cast<std::size_t>(i)]) {
    raw.push_back(grid.centre(unindex(static_cast<std::size_t>(i))));
  }
  std::reverse(raw.begin(), raw.end());
  raw.insert(raw.begin(), start);
  raw.push_back(goal);

  // Shorten: from each kept point, jump to the farthest later point that is in the clear.
  std::vector<Vec3> path{raw.front()};
  std::size_t i = 0;
  while (i + 1 < raw.size()) {
    std::size_t j = raw.size() - 1;
    while (j > i + 1 && !segment_clear(raw[i], raw[j], hazards)) --j;
    path.push_back(raw[j]);
    i = j;
  }
  return path;
}

}  // namespace astrum::minsnap
