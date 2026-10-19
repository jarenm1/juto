{
  description = "Juto Rust and GPUI development environment";

  nixConfig = {
    # Built by .github/workflows/cache.yml; keeps `nix profile install
    # github:jarenm1/juto` a download instead of a local Rust compile.
    extra-substituters = [ "https://juto.cachix.org" ];
    extra-trusted-public-keys = [
      "juto.cachix.org-1:uzMcDJsGbWJ1OXfgS34LOZI1GiZnLpd0xjXU54yMb2o="
    ];
  };

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts = {
      url = "github:hercules-ci/flake-parts";
      inputs.nixpkgs-lib.follows = "nixpkgs";
    };
    # Splits the Rust build: compiled dependencies live in a derivation keyed on
    # Cargo.lock + toolchain, so updates that only change our source recompile a
    # few crates instead of the whole graph.
    crane.url = "github:ipetkov/crane";
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" ];

      perSystem =
        { pkgs, config, ... }:
        let
          nativeLibraries = with pkgs; [
            fontconfig
            freetype
            zlib
            wayland
            libxkbcommon
            libxcb
            pixman
            libglvnd
            libgbm
            libdrm
            mesa
            vulkan-loader
            glib
            gst_all_1.gstreamer
            gst_all_1.gst-plugins-base
            gst_all_1.gst-plugins-bad
            gst_all_1.gst-plugins-good
            gst_all_1.gst-plugins-ugly
            gst_all_1.gst-libav
            libnice
          ];
          # Pin `.out`: default outputs can be `-bin` splits without the plugin .so files
          # (core `queue` lives in `gstreamer`'s main output; `libgstnice.so` in libnice's).
          gstPluginPath = pkgs.lib.makeSearchPath "lib/gstreamer-1.0" (
            map (pkg: pkg.out) (
              with pkgs;
              [
                gst_all_1.gstreamer
                gst_all_1.gst-plugins-base
                gst_all_1.gst-plugins-good
                gst_all_1.gst-plugins-bad
                gst_all_1.gst-plugins-ugly
                gst_all_1.gst-libav
                libnice
              ]
            )
          );
          # Xwayland is spawned per environment; xeyes exercises the X11 path.
          xwaylandTools = with pkgs; [
            xwayland
            xorg.xeyes
          ];
          # Each environment runs its own PipeWire instance for private audio.
          audioTools = with pkgs; [
            pipewire
          ];

          craneLib = inputs.crane.mkLib pkgs;

          # Only files Cargo reads; doc/script changes keep the same store path.
          jutoSrc = builtins.path {
            name = "juto-src";
            path = ./.;
            filter =
              path: type:
              let
                s = toString path;
                base = baseNameOf path;
                inDir = d: pkgs.lib.hasInfix "/${d}/" s;
              in
              # Directories are kept so the filter traverses them; files are
              # whitelisted, so unrelated trees (docs/, scripts/) stay empty.
              type == "directory"
              || pkgs.lib.elem base [
                "Cargo.toml"
                "Cargo.lock"
              ]
              || pkgs.lib.any inDir [
                ".cargo"
                "apps"
                "crates"
                "licenses"
              ];
          };

          commonArgs = {
            pname = "juto";
            version = "0.1.0";
            src = jutoSrc;
            strictDeps = true;
            nativeBuildInputs = [
              pkgs.clang
              pkgs.mold
              pkgs.makeWrapper
              pkgs.pkg-config
            ];
            buildInputs = nativeLibraries;
            doCheck = false;
          };

          cargoArtifacts = craneLib.buildDepsOnly commonArgs;

          juto = craneLib.buildPackage (
            commonArgs
            // {
              inherit cargoArtifacts;
              cargoExtraArgs = "-p juto --bin juto";
              doInstallCargoArtifacts = false;
              installPhaseCommand = ''
                mkdir -p $out/bin
                cp target/release/juto $out/bin/juto
              '';
              postFixup = ''
                wrapProgram $out/bin/juto \
                  --prefix LD_LIBRARY_PATH : "/run/opengl-driver/lib:${pkgs.lib.makeLibraryPath nativeLibraries}" \
                  --set GST_PLUGIN_SYSTEM_PATH_1_0 "${gstPluginPath}"
              '';
              meta = {
                description = "Native GPUI agent runtime";
                homepage = "https://github.com/jarenm1/juto";
                license = pkgs.lib.licenses.mit;
                mainProgram = "juto";
                platforms = [ "x86_64-linux" ];
              };
            }
          );
          # Shared compile cache for agent workspaces. Each workspace keeps its
          # own target/; sccache serves identical registry-crate compiles across
          # workspaces from one cache, capped by SCCACHE_CACHE_SIZE. sccache
          # refuses to run while CARGO_INCREMENTAL is set (inherited from an
          # older shell), so the hook unsets it. See docs/agent-workflow.md
          # "Build profile".
          compileCacheHook = ''
            unset CARGO_INCREMENTAL
            export SCCACHE_DIR="''${SCCACHE_DIR:-''${XDG_CACHE_HOME:-$HOME/.cache}/sccache}"
            export SCCACHE_CACHE_SIZE="''${SCCACHE_CACHE_SIZE:-10G}"
          '';
        in
        {
          packages.default = juto;
          packages.juto = juto;

          formatter = pkgs.nixfmt-tree;

          devShells.default = pkgs.mkShell {
            packages =
              with pkgs;
              [
                rustc
                cargo
                rustfmt
                clippy
                rust-analyzer
                clang
                mold
                pkg-config
                cmake
                ninja
                jujutsu
                git
                sccache
              ]
              ++ xwaylandTools
              ++ audioTools;
            buildInputs = nativeLibraries;

            RUST_SRC_PATH = "${pkgs.rustPlatform.rustLibSrc}";

            # Shared compile cache: sccache serves registry crates across agent
            # workspaces. Workspace members stay incremental and pass through.
            RUSTC_WRAPPER = "${pkgs.sccache}/bin/sccache";

            # GPUI loads graphics libraries at runtime. Use the host GPU driver.
            shellHook = compileCacheHook + ''
              export LD_LIBRARY_PATH="/run/opengl-driver/lib:${pkgs.lib.makeLibraryPath nativeLibraries}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
              export GST_PLUGIN_SYSTEM_PATH_1_0="${gstPluginPath}"
            '';
          };

          # Separate tools and a software GPU for native-window smoke checks.
          devShells.smoke = pkgs.mkShell {
            inputsFrom = [ config.devShells.default ];
            packages =
              with pkgs;
              [
                xvfb-run
                xdotool
                imagemagick
                python3
              ]
              ++ xwaylandTools
              ++ audioTools;
            LIBGL_ALWAYS_SOFTWARE = "1";
            RUSTC_WRAPPER = "${pkgs.sccache}/bin/sccache";
            shellHook = compileCacheHook;
            VK_DRIVER_FILES = "${pkgs.mesa}/share/vulkan/icd.d/lvp_icd.x86_64.json";
            FONTCONFIG_FILE = pkgs.makeFontsConf {
              fontDirectories = [ pkgs.dejavu_fonts ];
            };
          };
        };
    };
}
