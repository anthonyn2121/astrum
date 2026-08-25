# syntax=docker/dockerfile:1.7
#
# ASTRUM development image.
#   Stage 1 (jsbsim-builder): compile JSBSim as a shared library from source.
#   Stage 2 (dev):            full toolchain — Rust, C++/CMake, Zenoh, Cap'n Proto,
#                             Python — plus the JSBSim lib/headers/data.
#
# Nothing here is installed on the host; everything lives in this image.

ARG UBUNTU_VERSION=24.04

##############################################################################
# Stage 1 — build JSBSim (shared lib) from source
##############################################################################
FROM ubuntu:${UBUNTU_VERSION} AS jsbsim-builder

# JSBSim git ref (tag) and install prefix. Override with --build-arg if needed.
ARG JSBSIM_REF=v1.3.1
ARG JSBSIM_PREFIX=/opt/jsbsim
ENV DEBIAN_FRONTEND=noninteractive

# JSBSim needs only a C++17 compiler + CMake (XML via bundled Expat, no protobuf).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 --branch "${JSBSIM_REF}" https://github.com/JSBSim-Team/jsbsim.git .

# Configure -> build -> install the devel + runtime bits (headers, libJSBSim.so,
# JSBSim.pc, JSBSim exe). `make install` is no longer supported; use cmake --install.
RUN cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_INSTALL_PREFIX="${JSBSIM_PREFIX}" \
    && cmake --build build -j"$(nproc)" \
    && cmake --install build

# Stage the aircraft/engine/systems/scripts data as the runtime root directory
# (the path later passed to FGFDMExec::SetRootDir).
RUN mkdir -p "${JSBSIM_PREFIX}/share/jsbsim" \
    && cp -r aircraft engine systems scripts "${JSBSIM_PREFIX}/share/jsbsim/"

##############################################################################
# Stage 2 — ASTRUM dev toolchain
##############################################################################
FROM ubuntu:${UBUNTU_VERSION} AS dev

ARG JSBSIM_PREFIX=/opt/jsbsim
ARG RUST_CHANNEL=stable
# Zenoh release version — the C/C++ bindings (debs) and the Python wheel are all
# pinned here so they stay protocol-compatible with the Rust `zenoh` crate.
ARG ZENOH_VERSION=1.10.0
ENV DEBIAN_FRONTEND=noninteractive

# System toolchain + libraries (confirmed noble package names).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        g++ \
        cmake \
        git \
        pkg-config \
        curl \
        ca-certificates \
        capnproto \
        libcapnp-dev \
        python3 \
        python3-venv \
        python3-pip \
        gdb \
        less \
        vim-tiny \
    && rm -rf /var/lib/apt/lists/*

# Zenoh C/C++ bindings for the C++ controller. The Rust sim uses the pure-Rust
# `zenoh` crate and Python uses the eclipse-zenoh wheel, but the C++ controller
# links libzenohc (zenoh-c) through the zenoh-cpp header wrapper — both ship as
# prebuilt debs from the Eclipse Zenoh releases (zipped). Install runtime lib +
# dev bits (headers, cmake/pkgconfig, static lib) + the C++ headers, then
# ldconfig so libzenohc.so resolves from /usr/lib.
RUN cd /tmp \
    && curl -fsSL -o zenoh-c.zip \
       "https://github.com/eclipse-zenoh/zenoh-c/releases/download/${ZENOH_VERSION}/libzenohc-${ZENOH_VERSION}-x86_64-unknown-linux-gnu-debian.zip" \
    && curl -fsSL -o zenoh-cpp.zip \
       "https://github.com/eclipse-zenoh/zenoh-cpp/releases/download/${ZENOH_VERSION}/zenohcpp-${ZENOH_VERSION}-debian.zip" \
    && python3 -m zipfile -e zenoh-c.zip . \
    && python3 -m zipfile -e zenoh-cpp.zip . \
    && dpkg -i "libzenohc_${ZENOH_VERSION}_amd64.deb" \
               "libzenohc-dev_${ZENOH_VERSION}_amd64.deb" \
               "libzenohcpp-dev_${ZENOH_VERSION}_all.deb" \
    && ldconfig \
    && rm -f /tmp/zenoh-c.zip /tmp/zenoh-cpp.zip /tmp/*.deb

# JSBSim (headers, libJSBSim.so, JSBSim.pc, JSBSim exe, data) from the builder stage.
COPY --from=jsbsim-builder ${JSBSIM_PREFIX} ${JSBSIM_PREFIX}
ENV JSBSIM_PREFIX=${JSBSIM_PREFIX}
ENV PKG_CONFIG_PATH=${JSBSIM_PREFIX}/lib/pkgconfig
ENV LD_LIBRARY_PATH=${JSBSIM_PREFIX}/lib
ENV JSBSIM_ROOT=${JSBSIM_PREFIX}/share/jsbsim
# Register libJSBSim with the dynamic linker so it resolves without LD_LIBRARY_PATH too.
RUN echo "${JSBSIM_PREFIX}/lib" > /etc/ld.so.conf.d/jsbsim.conf && ldconfig

# Rust toolchain (rustup, system-wide under /usr/local).
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo
ENV PATH=/opt/venv/bin:${CARGO_HOME}/bin:${JSBSIM_PREFIX}/bin:${PATH}
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --no-modify-path --profile minimal --default-toolchain "${RUST_CHANNEL}" \
    && rustup component add rustfmt clippy \
    && chmod -R a+rwX "${CARGO_HOME}" "${RUSTUP_HOME}"

# Python venv for the Python controller and tooling. eclipse-zenoh is pinned to
# the same release as the Rust crate / C bindings for protocol compatibility.
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv "${VIRTUAL_ENV}" \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir "eclipse-zenoh==${ZENOH_VERSION}" pycapnp

# JSBSim 1.3.1 does not install a pkg-config file, so synthesize one. This lets
# astrum-fdm's build.rs and the C++ controller link JSBSim via `pkg-config JSBSim`
# instead of hardcoded paths. (Headers live under include/JSBSim; rpath baked in
# so linked binaries run without LD_LIBRARY_PATH.)
RUN mkdir -p "${JSBSIM_PREFIX}/lib/pkgconfig" && { \
      echo "prefix=${JSBSIM_PREFIX}"; \
      echo 'exec_prefix=${prefix}'; \
      echo 'libdir=${prefix}/lib'; \
      echo 'includedir=${prefix}/include/JSBSim'; \
      echo ''; \
      echo 'Name: JSBSim'; \
      echo 'Description: JSBSim flight dynamics model library'; \
      echo 'Version: 1.3.1'; \
      echo 'Libs: -L${libdir} -Wl,-rpath,${libdir} -lJSBSim'; \
      echo 'Cflags: -I${includedir}'; \
    } > "${JSBSIM_PREFIX}/lib/pkgconfig/JSBSim.pc"

WORKDIR /workspace
COPY docker/entrypoint.sh /usr/local/bin/astrum-entrypoint
RUN chmod +x /usr/local/bin/astrum-entrypoint
ENTRYPOINT ["/usr/local/bin/astrum-entrypoint"]
CMD ["bash"]
