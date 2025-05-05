rm -r out

python -m src.main HAM10000 sampling ham10000 gan \
--n-samples 50000 \
--alpha 0.99 \
--batch-norm \
--d-iters 1 \
--g-iters 1 \
--disc-h-layers 1 \
--disc-h-size -1 \
--disc-lr 2e-4 \
--disc-type "biggan" \
--eval-samples 100 \
--gan-arch "biggan" \
--gan-mode "wgan" \
--gp-weight 20 \
--grad-acc 1 \
--grad-clamp 0.01 \
--h-layers 2 \
--h-size 32 \
--lr 4e-4 \
--gpu 0 \
--max-epochs 10000 \
--max-lambda 0.01 \
--min-lambda 0.0001 \
--data-bs 256 \
--ncm-bs 256 \
--no-repr \
--rep-type "real" \
--custom-query \
--u-size 32 \
--scale-u-size \
--wandb \
--wandb-org-name "paulkroe" \
--wandb-project-name  "NeuralCausalAbstractions" \
# --scale-h-size \