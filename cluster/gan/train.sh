host="biwidl100"
user="$USER"
bd="/scratch_net/$host/$user/mammography"

qsub $bd/resources/biwi/run_on_host.sh MaskTrue_NoAugment_ICNR_nnUPSAMPLE_Lam0.0_SpectralNorm True True nn_upsample_conv 0.0 5000 True
qsub $bd/resources/biwi/run_on_host.sh MaskFalse_NoAugment_ICNR_nnUPSAMPLE_Lam0.0_SpectralNorm False True nn_upsample_conv 0.0 5000 True
