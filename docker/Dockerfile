# ASTRUM development image.
#   Stage 1 (jsbsim-builder): compile JSBSim as a shared library from source.
#   Stage 2 (dev):            full toolchain — Rust, C++/CMake, Zenoh, Cap'n Proto,
#                             Python — plus the JSBSim lib/headers/data.
#
# Nothing here is installed on the host; everything lives in this image. Every
# external input is pinned: the base image by digest, JSBSim by tag + commit,
# rustup-init / zenoh-c / zenoh-cpp / cargo-deny by version + sha256 (amd64 and
# arm64, selected by TARGETARCH), the Rust toolchain by exact version, and the
# Python packages by exact version (docker/requirements.txt).
#
# Processes run as the non-root user `astrum` (build args UID/GID, default
# 1000; `make build` passes the host user's ids) so files written into the bind
# mount stay owned by the host user. See docker/entrypoint.sh.

# ubuntu:24.04 (noble). Refresh with:
#   docker pull ubuntu:24.04 && docker inspect ubuntu:24.04 --format '{{index .RepoDigests 0}}'
ARG BASE_IMAGE=ubuntu:24.04@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517

##############################################################################
# Stage 1 — build JSBSim (shared lib) from source
##############################################################################
FROM ${BASE_IMAGE} AS jsbsim-builder

# JSBSim tag and the commit it MUST resolve to (guards against a moved tag),
# plus the install prefix. Override both together with --build-arg.
ARG JSBSIM_REF=v1.3.1
ARG JSBSIM_COMMIT=3b25f25e49b42d0489c04ac805674fc1450ca579
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
RUN git clone --depth 1 --branch "${JSBSIM_REF}" https://github.com/JSBSim-Team/jsbsim.git . \
    && actual="$(git rev-parse HEAD)" \
    && if [ "${actual}" != "${JSBSIM_COMMIT}" ]; then \
         echo "JSBSim ${JSBSIM_REF} resolved to ${actual}, expected ${JSBSIM_COMMIT}" >&2; exit 1; \
       fi

# Configure -> build -> install the devel + runtime bits (headers, libJSBSim.so,
# JSBSim exe). `make install` is no longer supported; use cmake --install.
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

# JSBSim does not install a pkg-config file, so synthesize one (version derived
# from JSBSIM_REF). This lets astrum-fdm's build.rs and the C++ controllers link
# JSBSim via `pkg-config JSBSim` instead of hardcoded paths. Headers live under
# include/JSBSim; the rpath is baked in so linked binaries run without
# LD_LIBRARY_PATH.
RUN mkdir -p "${JSBSIM_PREFIX}/lib/pkgconfig" && { \
      echo "prefix=${JSBSIM_PREFIX}"; \
      echo 'exec_prefix=${prefix}'; \
      echo 'libdir=${prefix}/lib'; \
      echo 'includedir=${prefix}/include/JSBSim'; \
      echo ''; \
      echo 'Name: JSBSim'; \
      echo 'Description: JSBSim flight dynamics model library'; \
      echo "Version: ${JSBSIM_REF#v}"; \
      echo 'Libs: -L${libdir} -Wl,-rpath,${libdir} -lJSBSim'; \
      echo 'Cflags: -I${includedir}'; \
    } > "${JSBSIM_PREFIX}/lib/pkgconfig/JSBSim.pc"

##############################################################################
# Stage 2 — ASTRUM dev toolchain
##############################################################################
FROM ${BASE_IMAGE} AS dev

# Set by BuildKit/buildx to the platform being built (amd64 | arm64).
ARG TARGETARCH
ARG JSBSIM_PREFIX=/opt/jsbsim
ENV DEBIAN_FRONTEND=noninteractive

# System toolchain + libraries (noble package names):
#   * libeigen3-dev       — header-only linear algebra for controllers/optimal/cpp
#   * libgtest-dev        — C++ unit tests (`make test-cpp`)
#   * nlohmann-json3-dev  — JSON for the C++ wire-fixture writer/verifier
#   * clang-format        — C++ formatter (`make fmt` / `make fmt-check`)
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
        libeigen3-dev \
        libgtest-dev \
        nlohmann-json3-dev \
        clang-format \
        python3 \
        python3-venv \
        gdb \
        less \
        vim-tiny \
    && rm -rf /var/lib/apt/lists/*

# Zenoh C/C++ bindings for the C++ controllers. The Rust side uses the pure-Rust
# `zenoh` crate and Python the eclipse-zenoh wheel; the C++ controllers link
# libzenohc (zenoh-c) through the zenoh-cpp header wrapper — both ship as
# prebuilt debs (zipped) from the Eclipse Zenoh releases. ZENOH_VERSION MUST
# match the Rust crate in Cargo.lock and eclipse-zenoh in docker/requirements.txt
# (scripts/verify-toolchain.sh fails otherwise). Checksums: sha256 of the zips.
ARG ZENOH_VERSION=1.10.0
ARG ZENOHC_SHA256_AMD64=b9a203795128a241f62268913206bea068773da1d5d0d0dbc7a9e7e412a0aa5f
ARG ZENOHC_SHA256_ARM64=c93c264ca632a0fc9be13ac58588080b459f82d36d27bc332e9d1ca91976b577
ARG ZENOHCPP_SHA256=fa4d9d5c59f1906affd68a51f3470842e95572802038e039b643e76881daab3b
RUN set -eu; \
    case "${TARGETARCH}" in \
      amd64) triple=x86_64-unknown-linux-gnu;  sha="${ZENOHC_SHA256_AMD64}" ;; \
      arm64) triple=aarch64-unknown-linux-gnu; sha="${ZENOHC_SHA256_ARM64}" ;; \
      *) echo "unsupported TARGETARCH '${TARGETARCH}' (amd64|arm64)" >&2; exit 1 ;; \
    esac; \
    cd /tmp; \
    curl -fsSL -o zenoh-c.zip \
      "https://github.com/eclipse-zenoh/zenoh-c/releases/download/${ZENOH_VERSION}/libzenohc-${ZENOH_VERSION}-${triple}-debian.zip"; \
    curl -fsSL -o zenoh-cpp.zip \
      "https://github.com/eclipse-zenoh/zenoh-cpp/releases/download/${ZENOH_VERSION}/zenohcpp-${ZENOH_VERSION}-debian.zip"; \
    printf '%s  %s\n' "${sha}" zenoh-c.zip "${ZENOHCPP_SHA256}" zenoh-cpp.zip | sha256sum -c -; \
    python3 -m zipfile -e zenoh-c.zip .; \
    python3 -m zipfile -e zenoh-cpp.zip .; \
    dpkg -i "libzenohc_${ZENOH_VERSION}_${TARGETARCH}.deb" \
            "libzenohc-dev_${ZENOH_VERSION}_${TARGETARCH}.deb" \
            "libzenohcpp-dev_${ZENOH_VERSION}_all.deb"; \
    ldconfig; \
    rm -f /tmp/zenoh-c.zip /tmp/zenoh-cpp.zip /tmp/*.deb

# JSBSim (headers, libJSBSim.so, JSBSim.pc, JSBSim exe, data) from the builder stage.
COPY --from=jsbsim-builder ${JSBSIM_PREFIX} ${JSBSIM_PREFIX}
ENV JSBSIM_PREFIX=${JSBSIM_PREFIX}
ENV PKG_CONFIG_PATH=${JSBSIM_PREFIX}/lib/pkgconfig
ENV LD_LIBRARY_PATH=${JSBSIM_PREFIX}/lib
ENV JSBSIM_ROOT=${JSBSIM_PREFIX}/share/jsbsim
# Register libJSBSim with the dynamic linker so it resolves without LD_LIBRARY_PATH too.
RUN echo "${JSBSIM_PREFIX}/lib" > /etc/ld.so.conf.d/jsbsim.conf && ldconfig

# Python venv for the Python controllers, the astrum-env adapter and tooling
# (ruff, pytest). The venv is read-only for the runtime user; nothing installs
# into it at run time. The two controller packages (astrum_client, astrum_optimal)
# are imported straight from the bind mount via PYTHONPATH.
ENV VIRTUAL_ENV=/opt/venv
COPY docker/requirements.txt /tmp/requirements.txt
RUN python3 -m venv "${VIRTUAL_ENV}" \
    && "${VIRTUAL_ENV}/bin/pip" install --no-cache-dir pip==26.2.2 \
    && "${VIRTUAL_ENV}/bin/pip" install --no-cache-dir --no-deps -r /tmp/requirements.txt \
    && "${VIRTUAL_ENV}/bin/pip" check \
    && rm /tmp/requirements.txt

# cargo-deny (`make deny`): static musl release binary, pinned + checksummed.
ARG CARGO_DENY_VERSION=0.20.2
ARG CARGO_DENY_SHA256_AMD64=9f12ed4c49936e09b48bf862b595cde2fe64fcbd9d74dfacac6131ca824c8d5f
ARG CARGO_DENY_SHA256_ARM64=995c82be0defc7a025cae49a2aa2644ce8245c9a3318fc4103907c6a285e8c7d
RUN set -eu; \
    case "${TARGETARCH}" in \
      amd64) arch=x86_64;  sha="${CARGO_DENY_SHA256_AMD64}" ;; \
      arm64) arch=aarch64; sha="${CARGO_DENY_SHA256_ARM64}" ;; \
      *) echo "unsupported TARGETARCH '${TARGETARCH}' (amd64|arm64)" >&2; exit 1 ;; \
    esac; \
    name="cargo-deny-${CARGO_DENY_VERSION}-${arch}-unknown-linux-musl"; \
    cd /tmp; \
    curl -fsSL -o deny.tar.gz \
      "https://github.com/EmbarkStudios/cargo-deny/releases/download/${CARGO_DENY_VERSION}/${name}.tar.gz"; \
    printf '%s  %s\n' "${sha}" deny.tar.gz | sha256sum -c -; \
    tar -xzf deny.tar.gz; \
    install -m 0755 "${name}/cargo-deny" /usr/local/bin/cargo-deny; \
    rm -rf deny.tar.gz "${name}"

# Runtime user. noble ships a `ubuntu` user at 1000:1000; drop it so the host's
# ids (often 1000) are free. --non-unique tolerates a GID that collides with a
# system group (e.g. macOS hosts' 20).
ARG UID=1000
ARG GID=1000
RUN userdel --remove ubuntu 2>/dev/null || true; \
    groupadd --non-unique --gid "${GID}" astrum \
    && useradd --non-unique --uid "${UID}" --gid "${GID}" --create-home --shell /bin/bash astrum

# Rust toolchain: rustup-init pinned + checksummed, toolchain pinned to the
# exact channel in rust-toolchain.toml (1.98.0; bump both together). Installed
# as `astrum` so CARGO_HOME (registry/git caches) and RUSTUP_HOME are writable
# by the runtime user.
ENV HOME=/home/astrum \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo
ENV PATH=${VIRTUAL_ENV}/bin:${CARGO_HOME}/bin:${JSBSIM_PREFIX}/bin:${PATH}
ARG RUST_TOOLCHAIN=1.98.0
ARG RUSTUP_VERSION=1.29.2
ARG RUSTUP_SHA256_AMD64=5be191e8c70a5007074254cb74d78dbfdd25cd5a61c27de7b8e5410122ec24f4
ARG RUSTUP_SHA256_ARM64=a9c168dbea81a84671c6b7a5c1795cb0eb699020110d107ddfca427bd797e697
RUN install -d -o astrum -g astrum "${CARGO_HOME}" "${RUSTUP_HOME}" /workspace /workspace/target
USER astrum
RUN set -eu; \
    case "${TARGETARCH}" in \
      amd64) triple=x86_64-unknown-linux-gnu;  sha="${RUSTUP_SHA256_AMD64}" ;; \
      arm64) triple=aarch64-unknown-linux-gnu; sha="${RUSTUP_SHA256_ARM64}" ;; \
      *) echo "unsupported TARGETARCH '${TARGETARCH}' (amd64|arm64)" >&2; exit 1 ;; \
    esac; \
    curl -fsSL -o /tmp/rustup-init \
      "https://static.rust-lang.org/rustup/archive/${RUSTUP_VERSION}/${triple}/rustup-init"; \
    printf '%s  %s\n' "${sha}" /tmp/rustup-init | sha256sum -c -; \
    chmod +x /tmp/rustup-init; \
    /tmp/rustup-init -y --no-modify-path --profile minimal \
      --default-toolchain "${RUST_TOOLCHAIN}" --component rustfmt,clippy; \
    rm /tmp/rustup-init; \
    mkdir -p "${CARGO_HOME}/registry" "${CARGO_HOME}/git"
USER root

# Tooling defaults for the bind-mounted workspace: controller packages
# importable from the checkout, and no caches/bytecode written into the repo.
ENV PYTHONPATH=/workspace/controllers/python:/workspace/controllers/optimal/python \
    PYTHONDONTWRITEBYTECODE=1 \
    RUFF_NO_CACHE=true \
    PYTEST_ADDOPTS="-p no:cacheprovider"

# Inside the container: `astrum` (scripts/bin/astrum — rebuild what changed,
# then run the CLI) is on PATH, and ASTRUM_DEV=1 makes `make` run its task
# targets in place instead of through `docker compose`.
ENV PATH=/workspace/scripts/bin:${PATH} \
    ASTRUM_DEV=1

# The interactive shell's look (`make dev` shells), Catppuccin Mocha: a
# powerline prompt (starship's catppuccin-powerline preset; its glyphs need a
# Nerd Font in the host terminal) and `ls` colours (vivid). Both release
# binaries, pinned + checksummed. docker/shell/bashrc wires them up from the
# end of ~/.bashrc; desktop notifications for long commands are off (no
# desktop in here).
ARG STARSHIP_VERSION=1.27.0
ARG STARSHIP_SHA256_AMD64=10c8eb4cb3b761d9bf59ea828444c2dab36c900b7d69a69e08814b286969df89
ARG STARSHIP_SHA256_ARM64=341f5cbf24699fa0b27ec88965ad25f862ea09e15d35a543fd6ffdff1d2aa00d
ARG VIVID_VERSION=0.11.1
ARG VIVID_SHA256_AMD64=7060e45ba1fc5b25d521704e65d2e5d92c1b219c06b7183ec120a87e1fc376d7
ARG VIVID_SHA256_ARM64=99bf6438b3403bc9255a83363fe5ed49a0f8bd86377a561420836af14dc8e9ee
RUN set -eu; \
    case "${TARGETARCH}" in \
      amd64) arch=x86_64;  starship_sha="${STARSHIP_SHA256_AMD64}"; vivid_sha="${VIVID_SHA256_AMD64}" ;; \
      arm64) arch=aarch64; starship_sha="${STARSHIP_SHA256_ARM64}"; vivid_sha="${VIVID_SHA256_ARM64}" ;; \
      *) echo "unsupported TARGETARCH '${TARGETARCH}' (amd64|arm64)" >&2; exit 1 ;; \
    esac; \
    cd /tmp; \
    curl -fsSL -o starship.tar.gz \
      "https://github.com/starship/starship/releases/download/v${STARSHIP_VERSION}/starship-${arch}-unknown-linux-musl.tar.gz"; \
    printf '%s  %s\n' "${starship_sha}" starship.tar.gz | sha256sum -c -; \
    tar -xzf starship.tar.gz starship; \
    install -m 0755 starship /usr/local/bin/starship; \
    vivid="vivid-v${VIVID_VERSION}-${arch}-unknown-linux-gnu"; \
    curl -fsSL -o vivid.tar.gz \
      "https://github.com/sharkdp/vivid/releases/download/v${VIVID_VERSION}/${vivid}.tar.gz"; \
    printf '%s  %s\n' "${vivid_sha}" vivid.tar.gz | sha256sum -c -; \
    tar -xzf vivid.tar.gz "${vivid}/vivid"; \
    install -m 0755 "${vivid}/vivid" /usr/local/bin/vivid; \
    rm -rf starship.tar.gz starship vivid.tar.gz "${vivid}"; \
    mkdir -p /etc/astrum; \
    starship preset catppuccin-powerline -o /etc/astrum/starship.toml; \
    STARSHIP_CONFIG=/etc/astrum/starship.toml starship config cmd_duration.show_notifications false; \
    echo '. /etc/astrum/bashrc' >> /home/astrum/.bashrc
COPY docker/shell/bashrc /etc/astrum/bashrc

WORKDIR /workspace
COPY docker/entrypoint.sh /usr/local/bin/astrum-entrypoint
RUN chmod 0755 /usr/local/bin/astrum-entrypoint
# The entrypoint starts as root only to hand the named volumes to `astrum`,
# then drops privileges for the actual command.
ENTRYPOINT ["/usr/local/bin/astrum-entrypoint"]
CMD ["bash"]
