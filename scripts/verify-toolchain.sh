#!/usr/bin/env bash
#
# Verify the ASTRUM toolchain image end-to-end. Run inside the dev container via
# `make verify`. Every check is hard (aborts on failure):
#   * pinned versions agree: rustc == rust-toolchain.toml; the Rust `zenoh`
#     crate (Cargo.lock) == zenoh-c/zenoh-cpp debs == eclipse-zenoh wheel ==
#     the astrum-client pyproject pin (Zenoh only guarantees protocol
#     compatibility within a release); the venv ==
#     docker/requirements.txt exactly;
#   * the tools exist (compilers, capnp, clang-format, ruff, pytest, cargo-deny);
#   * C++ programs link and run against libJSBSim and libzenohc, and the JSBSim
#     data root loads the c172x model.
set -euo pipefail
cd "$(dirname "$0")/.."

hr() { printf '%s\n' "============================================================"; }
fail() { echo "FAIL: $*" >&2; exit 1; }
same() { # same <what> <expected> <actual>
    [[ "$2" == "$3" ]] || fail "$1: expected '$2', found '$3'"
    printf '  %-28s %s\n' "$1" "$3"
}

hr
echo " ASTRUM toolchain verification"
hr

echo "-- tools --"
rustc --version
cargo --version
cargo fmt --version
cargo clippy --version
cargo deny --version
g++ --version | head -1
cmake --version | head -1
capnp --version
clang-format --version
python --version
ruff --version
python -m pytest --version

echo
echo "-- pinned versions agree --"
toolchain="$(python - <<'PY'
import tomllib
print(tomllib.load(open("rust-toolchain.toml", "rb"))["toolchain"]["channel"])
PY
)"
same "rustc == rust-toolchain.toml" "${toolchain}" "$(rustc --version | awk '{print $2}')"

zenoh_rust="$(python - <<'PY'
import tomllib
versions = {p["version"] for p in tomllib.load(open("Cargo.lock", "rb"))["package"] if p["name"] == "zenoh"}
if len(versions) != 1:
    raise SystemExit(f"Cargo.lock has zenoh versions {sorted(versions)}, expected exactly one")
print(versions.pop())
PY
)"
echo "  zenoh (Cargo.lock)           ${zenoh_rust}"
same "libzenohc deb" "${zenoh_rust}" "$(dpkg-query -W -f '${Version}' libzenohc)"
same "libzenohc-dev deb" "${zenoh_rust}" "$(dpkg-query -W -f '${Version}' libzenohc-dev)"
same "libzenohcpp-dev deb" "${zenoh_rust}" "$(dpkg-query -W -f '${Version}' libzenohcpp-dev)"
same "eclipse-zenoh wheel" "${zenoh_rust}" \
    "$(python -c 'import importlib.metadata as m; print(m.version("eclipse-zenoh"))')"
zenoh_pin="$(python - <<'PY'
import re
import tomllib
deps = tomllib.load(open("controllers/python/pyproject.toml", "rb"))["project"]["dependencies"]
pins = [m.group(1) for d in deps if (m := re.fullmatch(r"eclipse-zenoh\s*==\s*(\S+)", d))]
if len(pins) != 1:
    raise SystemExit(f"controllers/python/pyproject.toml must pin eclipse-zenoh==<v> once: {deps}")
print(pins[0])
PY
)"
same "astrum-client zenoh pin" "${zenoh_rust}" "${zenoh_pin}"

# The venv must hold exactly docker/requirements.txt (pip itself excluded).
python - <<'PY'
import importlib.metadata as m
import re

def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()

wanted = {}
for line in open("docker/requirements.txt"):
    line = line.split("#", 1)[0].strip()
    if line:
        name, version = line.split("==")
        wanted[norm(name)] = version
installed = {norm(d.metadata["Name"]): d.version for d in m.distributions()}
installed.pop("pip", None)
if installed != wanted:
    extra = sorted(set(installed.items()) - set(wanted.items()))
    missing = sorted(set(wanted.items()) - set(installed.items()))
    raise SystemExit(f"venv != docker/requirements.txt; unexpected {extra}, missing {missing}")
print(f"  {'venv == requirements.txt':<28} {len(wanted)} packages")
PY

echo
echo "-- libraries via pkg-config --"
echo "  capnp  : $(pkg-config --modversion capnp)"
echo "  JSBSim : $(pkg-config --modversion JSBSim)"

echo
echo "-- zenoh C/C++ bindings --"
# The shipped zenohc.pc has a stale /usr/local prefix, so check the actual
# installed artifacts: the runtime lib (dynamic linker) and the C++ header.
# (Not `ldconfig -p | grep -q`: grep exits at the first match, and under
# pipefail ldconfig's SIGPIPE on its next write fails the check.)
grep -q 'libzenohc\.so' <<<"$(ldconfig -p)" || fail "libzenohc.so not registered with ldconfig"
test -f /usr/include/zenoh.hxx || fail "/usr/include/zenoh.hxx missing"
echo "  libzenohc.so registered; /usr/include/zenoh.hxx present"

echo
echo "-- python bindings --"
python - <<'PY'
import capnp  # noqa: F401
import numpy  # noqa: F401
import scipy  # noqa: F401
import zenoh  # noqa: F401
print("  zenoh, capnp, numpy, scipy import OK")
PY

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

echo
echo "-- C++ links libJSBSim (compile + run smoke) --"
g++ -std=c++17 docker/checks/jsbsim_smoke.cpp $(pkg-config --cflags --libs JSBSim) -o "${tmp}/jsbsim_smoke"
"${tmp}/jsbsim_smoke"

echo
echo "-- C++ links libzenohc via zenoh-cpp (compile + run smoke) --"
g++ -std=c++17 -DZENOHCXX_ZENOHC docker/checks/zenoh_smoke.cpp -o "${tmp}/zenoh_smoke" -lzenohc -pthread
"${tmp}/zenoh_smoke"

echo
echo "-- JSBSim CLI + aircraft data --"
jsbsim_version="$(JSBSim --version 2>&1)"
grep -m1 -i 'version' <<<"${jsbsim_version}" || fail "JSBSim --version printed no version"
test -f "${JSBSIM_ROOT}/aircraft/c172x/c172x.xml" || fail "c172x missing under JSBSIM_ROOT=${JSBSIM_ROOT}"
echo "  c172x present under JSBSIM_ROOT=${JSBSIM_ROOT}"

echo
hr
echo " ALL CHECKS PASSED"
hr
