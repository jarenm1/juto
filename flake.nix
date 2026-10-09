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
        in
        {
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
