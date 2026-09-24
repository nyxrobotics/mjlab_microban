# Audited XC330 velocity bootstrap

`model_14999.pt` is the fixed actor source used by
`scripts/train_microban_teleop_v4_canary.sh`. It is stored in the repository so
the v4 canary starts from identical bytes on a fresh checkout.

```text
SHA-256 b0bcdadac39716be784207dd6b2b93157162a3e80650e23c05f490c400b9e141
source run mjlab_microban_velocity/2026-09-22_02-54-27/model_14999.pt
completed PPO updates 15000
```

The teleop bootstrap loader copies only the audited actor MLP and shared
63-input normalizer fields. It does not copy the critic, optimizer, action
distribution, or PPO iteration. The canary script verifies this digest before
loading the checkpoint and fails closed on any mismatch.
