{
  description = "A neural network written in Python using NumPy";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python313;
        pythonPackages = python.pkgs;

        cliffordEnv = python.withPackages (ps: with ps; [
          numpy
          pyside6
          toml
          tqdm
          pytest
          pytest-cov
          black
          ruff
          mypy
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            cliffordEnv
            pkgs.sqlite
          ];

          shellHook = ''
            export PYTHONPATH="${self}/src:$PYTHONPATH"
            export CLIFFORD_DB_PATH="${self}/database/clifford.db"
            export CLIFFORD_ASSET_PATH="${self}/assets/clifford.png"
            echo "Clifford development environment loaded"
            echo "Python: $(python --version)"
            echo "NumPy: $(python -c 'import numpy; print(numpy.__version__)')"
          '';
        };

        packages.default = cliffordEnv;

        apps.default = {
          type = "app";
          program = "${cliffordEnv}/bin/python";
        };
      }
    );
}
