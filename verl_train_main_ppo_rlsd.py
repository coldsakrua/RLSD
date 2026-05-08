from verl.trainer import main_ppo

from verl_rlsd_patch import patch_verl_compute_advantage


def main():
    patch_verl_compute_advantage()
    main_ppo.main()


if __name__ == "__main__":
    main()
