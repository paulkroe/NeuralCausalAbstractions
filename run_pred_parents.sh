python -m src.main CIFAR sampling cifar gan \
    --max-epochs 100 \
    --patience 100 \
    --h-layers 3 \
    --h-size 2 \
    --scale-h-size \
    --scale-u-size \
    --batch-norm \
    --img-size 32 \
    --data-bs 512 \
    --rep-bs 512 \
    --gan-mode wgan \
    --gan-arch biggan \
    --rep-size 1024 \
    --rep-image-only \
    --rep-h-layers 5 \
    --rep-h-size 128 \
    --rep-max-epochs 25 \
    --rep-patience 25 \
    --gpu 0 \
    --eval-samples 50 \
    --wandb \
    --wandb-project-name NeuralCausalAbstractions \
    --wandb-org-name paulkroe \
    --rep-pred-parents