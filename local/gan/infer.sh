mask_chkpt="40000"
nomask_chkpt="25000"
exp="NoAugment_ICNR_nnUPSAMPLE_Lam0.0_SpectralNorm"

for stage in "train" "eval" "test"; do
    # Mask
    qsub ./local/infer_on_host.sh "data_out/MaskTrue_$exp/chook/model.ckpt-${mask_chkpt}" True True nn_upsample_conv 0.0 "data_out/MaskTrue_${exp}_${mask_chkpt}_steps_inference_$stage" "$stage" True
    # No-Mask
    qsub ./local/infer_on_host.sh "data_out/MaskFalse_$exp/chook/model.ckpt-${nomask_chkpt}" False True nn_upsample_conv 0.0 "data_out/MaskFalse_${exp}_${nomask_chkpt}_steps_inference_$stage" "$stage" True
done
