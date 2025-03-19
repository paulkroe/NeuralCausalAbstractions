python -m src.main CIFAR sampling cifar gan \
    --max-epochs 200 \
    --patience 200 \
    --h-layers 3 \
    --h-size 2 \
    --scale-h-size \
    --scale-u-size \
    --batch-norm \
    --img-size 32 \
    --data-bs 256 \
    --rep-bs 256 \
    --gan-mode wgan \
    --gan-arch biggan \
    --rep-size 512 \
    --rep-image-only \
    --rep-h-layers 5 \
    --rep-h-size 128 \
    --rep-max-epochs 1 \
    --rep-patience 1 \
    --gpu 0 \
    --wandb \
    --wandb-project-name NeuralCausalAbstractions \
    --wandb-org-name paulkroe \
    --rep-pred-parents