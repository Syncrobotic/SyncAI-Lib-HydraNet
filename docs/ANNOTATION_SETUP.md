# Annotation environment

CVAT runs on a Google Cloud VM with **no public IP**. It is reached through an IAP tunnel,
so there is no login page exposed to the internet and no TLS certificate to maintain — the
tunnel is the encrypted channel, and Google IAM is the authentication.

Site footage is likely to show customer premises, which is why it is set up this way.

## What exists

| | |
|---|---|
| VM | `hydranet-annotation`, `asia-east1-b`, `e2-standard-2`, 100 GB |
| Project | `syncrobotic-aisw` |
| Public IP | none — IAP only |
| Firewall | `allow-iap-ssh-hydranet`: tcp:22 from `35.235.240.0/20` (Google's IAP range) to tagged hosts only |
| Storage | `gs://syncai-hydranet`, reachable from the VM over Private Google Access |
| Service account | `syncai-hydranet@…`, scoped to that one bucket |

## Connecting

Each annotator needs the `gcloud` CLI and IAM access to the project.

```bash
gcloud auth login
gcloud config set project syncrobotic-aisw

# open the tunnel; leave this running
gcloud compute ssh hydranet-annotation --zone=asia-east1-b \
    --tunnel-through-iap -- -L 8080:localhost:8080
```

Then open <http://localhost:8080> in a browser. The connection is local to your machine;
CVAT itself is never exposed.

The IAM roles an annotator needs are `roles/iap.tunnelResourceAccessor` and
`roles/compute.instanceAdmin.v1` (or a narrower custom role granting
`compute.instances.get` and `compute.instances.osLogin`).

## First-time setup

The admin account has to be created once, interactively, by whoever owns the instance:

```bash
gcloud compute ssh hydranet-annotation --zone=asia-east1-b --tunnel-through-iap
sudo docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```

Pick the password with a password manager and store it there. Annotator accounts are then
created from CVAT's own admin UI.

## Stop the VM when nobody is annotating

Annotation is bursty, and a stopped VM costs only its disk.

```bash
gcloud compute instances stop  hydranet-annotation --zone=asia-east1-b   # ~$10/month
gcloud compute instances start hydranet-annotation --zone=asia-east1-b   # ~$59/month running
```

CVAT restarts with the machine; the containers are configured to come back up. Give it a
couple of minutes before the API answers.

## The annotation spec

Everything below is a contract between annotators and training. If the two disagree, the
labels are silently wrong and no metric will reveal it.

- **Label the 12 indoor terrain classes**, ids exactly as in
  [`label_maps_indoor.py`](../src/syncai_hydranet/data/label_maps_indoor.py):

  | id | class | | id | class |
  |---|---|---|---|---|
  | 0 | `void` | | 6 | `threshold_ramp` |
  | 1 | `floor_hard` | | 7 | `wall` |
  | 2 | `floor_soft` | | 8 | `glass` |
  | 3 | `floor_metal` | | 9 | `door` |
  | 4 | `wet_slippery` | | 10 | `obstacle_furniture` |
  | 5 | `stairs` | | 11 | `person` |

- **Do not annotate traversability.** It is derived from terrain through the policy table
  in the same file. One annotation pass supervises both heads.
- **Ambiguous pixels get 255 (ignore), never a guess.** An ignore region costs nothing in
  training; a wrong label is trained on.
- Export as **single-channel PNG** where the pixel value *is* the class id, into the
  `seg_folder` layout, and set `label_map: indoor_native` in the config — the ids need no
  translation.

### What to prioritise

These three classes have **zero** examples in ADE20K and are the reason `caution` cannot
exceed about 0.23 no matter how long training runs:

1. `wet_slippery` — standing water, freshly mopped floor, spills
2. `floor_metal` — grating, steel plate, drain covers
3. `threshold_ramp` — door sills, ramps, level changes

Then `glass`, which has data but the highest cost when wrong: it reads as an open corridor.

Everything else — floor, wall, door, furniture, person — is adequately covered already.
Annotating more of those is close to wasted effort.

### Capture and split rules

- Shoot at **robot camera height and pitch**, not eye level. ADE20K's weakness is exactly
  this, and it is what makes the model worse on a robot than the numbers suggest.
- Vary lighting deliberately, and include the hard cases: backlit entrances, glare on
  polished floors, mirrors, glass with objects visible behind it.
- **Record continuous sessions and log which is which.** Session identity is what makes an
  honest train/val/test split possible; frames without provenance cannot be split safely.
- **Split by session, never randomly.** Adjacent frames are near-identical, so a random
  split measures memorisation and will look excellent while meaning nothing.

See [METHODOLOGY.md](METHODOLOGY.md) for how this fits the wider process, and
[TRAINING_GUIDE.md](TRAINING_GUIDE.md) for why the class list is shaped this way.
