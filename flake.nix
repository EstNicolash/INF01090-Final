{
  description = "Maritime Piracy Data Science – development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-darwin" ] (system:
      let
        pkgs = import nixpkgs { inherit system; };
        
        datasetUrl1 = "https://raw.githubusercontent.com/newzealandpaul/Maritime-Pirate-Attacks/master/data/csv/pirate_attacks.csv";
        datasetUrl2 = "https://raw.githubusercontent.com/newzealandpaul/Maritime-Pirate-Attacks/master/data/csv/country_codes.csv";
        datasetUrl3 = "https://raw.githubusercontent.com/newzealandpaul/Maritime-Pirate-Attacks/master/data/csv/country_indicators.csv";
        
        python = pkgs.python314;

        nativeBuildDeps = with pkgs; [
          stdenv.cc.cc.lib
          zlib
          cmake
          pkg-config
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          name = "maritime-piracy-ds";

          packages = [ python ] ++ nativeBuildDeps ++ (with pkgs; [
            curl
            python.pkgs.pip
            python.pkgs.virtualenv
          ]);

          shellHook = ''
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:$LD_LIBRARY_PATH"

            DATASET_URL="${datasetUrl1}"
            DATA_FILE="$(pwd)/data/raw/piracy_attacks.csv"
            echo "[nix] Downloading data…"
            if [ ! -f "$DATA_FILE" ]; then
              mkdir -p "$(dirname "$DATA_FILE")"
              if ! curl -sSL "$DATASET_URL" -o "$DATA_FILE"; then
                echo "[nix] Warning: failed to download piracy_attacks.csv (check network connectivity or dataset URL)"
              fi
            fi

            DATASET_URL="${datasetUrl2}"
            DATA_FILE="$(pwd)/data/raw/country_codes.csv"

            if [ ! -f "$DATA_FILE" ]; then
              mkdir -p "$(dirname "$DATA_FILE")"
              if ! curl -sSL "$DATASET_URL" -o "$DATA_FILE"; then
                echo "[nix] Warning: failed to download country_codes.csv (check network connectivity or dataset URL)"
              fi
            fi

            DATASET_URL="${datasetUrl3}"
            DATA_FILE="$(pwd)/data/raw/country_indicators.csv"

            if [ ! -f "$DATA_FILE" ]; then
              mkdir -p "$(dirname "$DATA_FILE")"
              if ! curl -sSL "$DATASET_URL" -o "$DATA_FILE"; then
                echo "[nix] Warning: failed to download country_indicators.csv(check network connectivity or dataset URL)"
              fi
            fi

            VENV_DIR="$(pwd)/.venv"

            if [ ! -d "$VENV_DIR" ]; then
              echo "[nix] Creating virtual environment at $VENV_DIR …"
              ${python}/bin/python -m venv "$VENV_DIR"
            fi

            source "$VENV_DIR/bin/activate"

            echo "[nix] Upgrading pip …"
            pip install --quiet --upgrade pip

            if [ -f "$(pwd)/requirements.txt" ]; then
              echo "[nix] Installing requirements.txt …"
              pip install --quiet -r "$(pwd)/requirements.txt"
            fi

            echo "[nix] Development environment ready. Python: $(python --version)"
          '';
        };
      });
}
