{
  description = "Juto Rust and GPUI development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts = {
      url = "github:hercules-ci/flake-parts";
      inputs.nixpkgs-lib.follows = "nixpkgs";
    };
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
            vulkan-loader
          ];
          juto = pkgs.rustPlatform.buildRustPackage {
            pname = "juto";
            version = "0.1.0";
            src = pkgs.lib.cleanSourceWith {
              src = ./.;
              filter =
                path: type:
                let
                  name = pkgs.lib.baseNameOf path;
                in
                name != ".git" && name != ".jj" && name != "target";
            };
            cargoLock.lockFile = ./Cargo.lock;
            cargoBuildFlags = [
              "-p"
              "juto"
              "--bin"
              "juto"
            ];
            cargoInstallFlags = [
              "-p"
              "juto"
              "--bin"
              "juto"
            ];
            nativeBuildInputs = [
              pkgs.clang
              pkgs.mold
              pkgs.makeWrapper
              pkgs.pkg-config
            ];
            buildInputs = nativeLibraries;
            doCheck = false;
            postFixup = ''
              wrapProgram $out/bin/juto \
                --prefix LD_LIBRARY_PATH : "/run/opengl-driver/lib:${pkgs.lib.makeLibraryPath nativeLibraries}"
            '';
            meta = {
              description = "Native GPUI agent runtime";
              homepage = "https://github.com/jarenm1/juto";
              license = pkgs.lib.licenses.mit;
              mainProgram = "juto";
              platforms = [ "x86_64-linux" ];
            };
          };
        in
        {
          packages.default = juto;
          packages.juto = juto;

          formatter = pkgs.nixfmt-tree;

          devShells.default = pkgs.mkShell {
            packages = with pkgs; [
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
            ];
            buildInputs = nativeLibraries;

            RUST_SRC_PATH = "${pkgs.rustPlatform.rustLibSrc}";

            # Keep Cargo's incremental cache usable; do not wrap rustc with sccache.
            CARGO_INCREMENTAL = "1";

            # GPUI loads graphics libraries at runtime. Use the host GPU driver.
            shellHook = ''
              export LD_LIBRARY_PATH="/run/opengl-driver/lib:${pkgs.lib.makeLibraryPath nativeLibraries}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            '';
          };

          # Separate tools and a software GPU for native-window smoke checks.
          devShells.smoke = pkgs.mkShell {
            inputsFrom = [ config.devShells.default ];
            packages = with pkgs; [
              xvfb-run
              xdotool
              imagemagick
              python3
            ];
            LIBGL_ALWAYS_SOFTWARE = "1";
            VK_DRIVER_FILES = "${pkgs.mesa}/share/vulkan/icd.d/lvp_icd.x86_64.json";
            FONTCONFIG_FILE = pkgs.makeFontsConf {
              fontDirectories = [ pkgs.dejavu_fonts ];
            };
          };
        };
    };
}
