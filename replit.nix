# Nix environment for AURUM on Replit
# Provides Python 3.11 and system-level C libraries required by
# pandas-ta (via numpy/scipy), vectorbt, and scikit-learn.
{ pkgs }: {
  deps = [
    # Python 3.11 interpreter
    pkgs.python311Full
    pkgs.python311Packages.pip

    # C/C++ toolchain — needed to compile native extensions
    # (numpy, scipy, pandas, scikit-learn wheels may build from source)
    pkgs.gcc
    pkgs.gnumake
    pkgs.cmake

    # Linear algebra backends for numpy / scipy / vectorbt
    pkgs.openblas
    pkgs.lapack

    # HDF5 — optional, used by pandas for .h5 caching
    pkgs.hdf5

    # System libs often required by data-science Python packages
    pkgs.zlib
    pkgs.libffi
    pkgs.openssl
    pkgs.readline

    # pkg-config to help pip find native dependencies
    pkgs.pkg-config
  ];

  env = {
    # Tell numpy / scipy where to find BLAS/LAPACK at build time
    OPENBLAS = "${pkgs.openblas}/lib";
    LD_LIBRARY_PATH = "${pkgs.openblas}/lib:${pkgs.zlib}/lib";
  };
}
