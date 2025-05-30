rm -r out

python -m src.main HAM10000 sampling ham10000 gan \
--n-samples 10000 \
--alpha 0.99 \
--batch-norm \
--d-iters 1 \
--g-iters 1 \
--disc-h-layers 2 \
--disc-h-size -1 \
--disc-lr 1e-4 \
--disc-type "standard" \
--eval-samples 100 \
--feature-maps 64 \
--gan-arch "dcgan" \
--gan-mode "wgan" \
--gp-weight 10 \
--grad-acc 1 \
--grad-clamp 0.01 \
--h-layers 2 \
--h-size 1 \
--lr 1e-4 \
--gpu 0 \
--max-epochs 1000 \
--max-lambda 0.01 \
--min-lambda 0.0001 \
--data-bs 512 \
--ncm-bs 512 \
--no-repr \
--patience 250 \
--rep-type "real" \
--custom-query \
--scale-h-size \
--u-size 1 \
--wandb \
--wandb-org-name "paulkroe" \
--wandb-project-name  "NeuralCausalAbstractions"
# --scale-u-size \