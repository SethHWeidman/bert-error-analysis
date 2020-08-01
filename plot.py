import os
import pickle

import matplotlib.pyplot as plt

import const
import train

BASE_PLOT_PATH = os.path.join(const.BASE_DIR, 'plots')

def plot_losses() -> None:
    mlm_losses, nsp_losses = get_most_recent_mlm_nsp_losses()

    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 7), constrained_layout=True)

    f.suptitle(
        """
        Average loss per observation over batches on masked language 
        modeling and next sentence prediction tasks (50 epochs)
        """
    )

    ax1.plot(mlm_losses)
    _, ymax = ax1.get_ylim()
    ax1.set_ylim(0, ymax)
    ax1.set_title('Masked language model loss over batches')

    ax2.plot(nsp_losses)
    _, ymax = ax2.get_ylim()
    ax2.set_ylim(0, ymax)
    ax2.set_xlabel("Batch")
    ax2.set_title('Next sentence prediction loss over batches')

    f.savefig(os.path.join(BASE_PLOT_PATH, "loss_over_epochs.png"))
    plt.show()


def get_most_recent_dir() -> str:
    return max(os.listdir(train.BASE_LOG_PATH))


def get_most_recent_mlm_nsp_losses():
    return (
        pickle.load(
            open(
                os.path.join(train.BASE_LOG_PATH, get_most_recent_dir(), 'batch_losses_mlm.p'),
                'rb',
            )
        ),
        pickle.load(
            open(
                os.path.join(train.BASE_LOG_PATH, get_most_recent_dir(), 'batch_losses_nsp.p'),
                'rb',
            )
        ),
    )


if __name__ == '__main__':
    plot_losses()
