# fix3 통합 실험 설계

## 목적

R8 위에 보정 모듈을 더 붙이는 방식 대신, 지금까지 반복해서 관측된 구조적 원인을 한 번에 제거한다. 이 실험은 묶음 전체가 R8의 held-out PSNR 15.39 dB를 넘는지를 검증한다. 여러 변경을 동시에 적용하므로 개별 변경의 인과 기여도는 이 런 하나로 분리할 수 없다.

## 적용 사항

- `pruning_scores`를 샘플 선택에서 완전히 제거했다. opacity와 공간 밀도처럼 입력 Gaussian 자체에서 계산되는 값만 쓴다.
- step 12000 미만의 미성숙 snapshot을 train/val 모두에서 제외한다. 실제 split은 train 1628, val 173개가 남는다.
- 고정 scene anchor와 8-nearest local spill을 사용한다. F3에서 사용했던 전역 `capacitated` 보정은 빈 용량을 채우려고 점을 먼 anchor로 이동시켜 지역 그룹을 깨뜨릴 수 있으므로 F3B에서는 사용하지 않는다.
- compressor의 shape 19 / appearance 8 독립 head를 하나의 27채널 head로 바꿨다.
- J1의 zero-init residual refiner와 기존 geometry/attribute decoder 조합을 쓰지 않는다. 동일한 shared-27 slot representation이 xyz, scale, quaternion, opacity, color, presence를 직접 출력한다.
- encoder와 decoder가 동일한 고정 scene anchor를 그룹 원점으로 사용한다. 입력 patch에는 `point - fixed_anchor`를 넣어 target group 중심 오프셋 정보가 보존되며, decoder는 별도의 bounded group translation을 예측한다. 따라서 F3처럼 decoder 중심이 encoder 표본 중심에 사실상 고정되지 않는다.
- xyz는 `fixed_anchor + learned_translation + centred_local_shape * scale`로 구성한다. local shape은 non-collapsed Fibonacci basis에서 시작하고 shared code가 anisotropic frame과 bounded deformation을 예측한다. scale은 group-relative base와 비대칭 band를 사용해 giant-splat 우회를 제한한다.
- intra-group Chamfer/Sinkhorn은 예측과 GT의 중심을 각각 제거하고 **GT group extent**로 정규화하여 순수 형상만 감독한다. group centroid loss는 extent로 나누지 않은 절대 좌표 오차로 translation을 직접 감독한다. 이로써 중심 오차를 local shape 확대가 대신 갚던 F3의 충돌을 제거한다.
- balanced global Chamfer, coverage, centroid, centred intra-group loss, presence를 위치 감독으로 사용한다. slot-to-slot L1과 responsibility 기반 scale 확장은 사용하지 않는다.
- 학습은 `0–1500 geometry/presence 전용 → 1500–2500 attribute parameter ramp → 2500 이후 render ramp` 순서다. 속성이 shared-27을 건드리기 전에 xyz가 먼저 자리 잡으며, geometry 손실은 이후에도 계속 유지한다.
- 렌더 loss에서는 xyz를 항상 detach한다. 이미지 그래디언트로 3D 위치를 운반하지 않지만, scale/rotation/opacity/color 경로를 통해 같은 shared-27 전체에는 그래디언트가 간다.
- attribute parameter anchor는 render 시작 후 1500 step 동안 0으로 감쇠하며, predicted presence를 실제 렌더에 사용한다.
- 한 방문마다 fresh real view 1개를 추가한다. 5→10 view T4처럼 한 step의 view 수만 늘리는 방식은 사용하지 않는다.
- `max_points=262144`, `z_compact=32x64x64`, K와 latent 크기는 바꾸지 않는다. generative branch도 끈다.

## 실행과 판정

실행은 `CUDA_VISIBLE_DEVICES=1 bash scripts/launch_fix3_all_joint.sh --detached`이다. 100/250/500/1000 step에서 geometry를 보고, 특히 `cen`, `tr`, centred `ich/ot`, eval의 `cen`, `relq`, template effective rank가 함께 개선되는지 확인한다. 1500에서 attribute 전환, 2500에서 render 전환을 확인한다. 핵심 판정은 3000 step 이후 held-out photo PSNR, predicted-presence render, Gaussian scale 분포, nn_unique, empty fraction이다. 최종 성공 기준은 R8 15.39 dB 초과이며 19–20 dB는 mature GT render 약 20.13 dB에 접근하는 장기 목표다.

## F3B 판정과 F3C 구조 변경

F3B step 500에서 centroid RMSE는 0.03982(F3)에서 0.00585로 줄었지만,
`template_erank=3.1/97.6`, `nn_unique=0.489`, `rel_offset=1.288`이었다.
고정 anchor로 중심 표현 문제는 줄었지만 동일한 저차원 local template을 그룹마다
반복하는 문제는 남았다. 원인은 direct decoder가 compact cell의 learned 27채널을
그룹당 한 벡터로 만든 뒤 64개 query에 broadcast했고, decompressor가 복원한 그룹당
8개 local token은 완전히 무시했으며 direct 모드에서 decompressor까지 freeze했다는
점이다.

F3C는 compact latent 크기 `32x64x64`를 바꾸지 않고 다음과 같이 출력 경로를
계층화한다.

`z_compact[group,32] -> trainable decompressor -> local memory[8,32]`

`point query[64] --cross-attention--> local memory[8] -> xyz/scale/rot/opacity/color`

각 point query에는 Fibonacci 좌표 embedding과 slot embedding을 함께 주며, 8개
memory token은 별도의 positional embedding과 self-attention으로 contextualise한다.
geometry loss는 이 경로를 통해 decompressor, compressor, encoder까지 전달된다.
로그의 `mem`은 local memory token 간 표준편차이고 `d[deco ...]`는 decompressor가
실제로 움직였는지를 나타낸다. F3C의 1-step 실데이터 smoke test에서는
`mem=0.381`, `d[deco]=8e-05`, 6.9GB로 확인됐다.

F3C의 geometry gate는 step 500/1000에서 `template_erank`, `nn_unique`, `relq`이다.
중심 RMSE만 좋아지고 이 세 지표가 개선되지 않으면 attribute/render 단계를 켜지
않는다.

## F3C 판정과 F3D compact 구조 변경

F3C step 500은 `template_erank=3.59/97.62`, `nn_unique=0.508`,
`relq=1.307`로 세 geometry gate를 모두 통과하지 못했다. 8개 reconstructed
memory token은 서로 달라졌지만, compressor가 먼저 encoder의 8개 토큰을 하나의
group vector로 합친 뒤 27채널을 출력했기 때문에 decoder의 8개 토큰은 이미 잃은
정보를 다시 펼친 것에 불과했다.

F3D는 compact 크기 `32x64x64`와 전체 learned budget 27을 그대로 유지하면서
non-anchor 채널의 의미를 다음처럼 고정한다.

`global[3] | local_0[6] | local_1[6] | local_2[6] | local_3[6]`

4개의 learned pooling query가 compressor의 `8 x model_dim` encoder token에
직접 cross-attention하고, 각각의 결과가 별도 6채널 local code로 투영된다.
historical `8 tokens -> one group_vec -> 27 channels` 경로를 local code가 우회한다.
decoder는 24채널을 그대로 `4 x 6` memory로 reshape하며, global 3채널만 그룹 및
이웃 context에 사용한다. 따라서 local 정보는 point-query cross-attention 이외의
broadcast 경로로 새지 않는다. F3D에서는 decompressor가 출력 경로에 없고 freeze된
것이 의도된 동작이다.

로그의 `cmem`은 compact에 실제 기록된 네 local code 사이의 표준편차이고 `mem`은
decoder projection 이후의 memory 다양성이다. 262k one-step smoke test는
`cmem=0.033`, `mem=0.034`, encoder/compressor/joint decoder 모두 non-zero 이동,
7.2GB로 통과했다.
