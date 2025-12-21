import os
import sys
import glob
import traceback

def main():
    try:
        import torch
        p = os.path.dirname(torch.__file__)
        print("torch.__file__:", torch.__file__)
        libdir = os.path.join(p, "lib")
        print("torch libdir:", libdir)
        if os.path.exists(libdir):
            print("\nfiles in torch libdir:")
            for f in sorted(glob.glob(os.path.join(libdir, "*"))):
                print(f)
        else:
            print("lib dir not found:", libdir)

        fbgemm = os.path.join(libdir, "fbgemm.dll")
        print("\nfbgemm exists:", os.path.exists(fbgemm))
        if os.path.exists(fbgemm):
            print("fbgemm path:", fbgemm)

        # Also print site-packages torch lib folder listing as fallback
        site_lib = os.path.join(os.path.dirname(torch.__file__))
        print("\nsite-packages torch folder:", site_lib)
        for f in sorted(glob.glob(os.path.join(site_lib, "*.dll"))):
            print(f)

    except Exception:
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
