# Changelog

## [0.4.0](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/compare/v0.3.0...v0.4.0) (2026-09-03)


### Features

* **bev3d:** the scene vocabulary gains stairs and glass ([b3a64d3](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/b3a64d3902cabeb5e57934d1653fdb5e712b1b7a))


### Documentation

* stage-0 measured against the SpatialLM recipe, Cosmos placed ([4261e2c](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/4261e2c40d82d37b68ce9791e52c436d94876f67))
* the Cosmos night probe brackets the dial ([bab85c4](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/bab85c422128ef165057e9ffa166e46095153726))
* the metro and mall probe venues are built as 3D scenes ([90f551c](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/90f551c10a787a6d1841d3df44c96b70629d8492))
* the night sweep lands on D, judged against real fleet IR ([010eb1d](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/010eb1db8f5dbd2195995f17d7abcc57dbf13683))
* the shipped model measured in metro, mall and airport venues ([a6bdfd6](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/a6bdfd6c47268cf76a859cc124b9156d9ebe5410))
* the SpatialLM probe ran and measured its own answer ([5ebe13c](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/5ebe13ceb94734fa045a95d13057f9b56070ca86))
* the three foreign venues asked to self-calibrate, and two answer ([94024dd](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/94024dd45a6275ac2d486735935591246ebc54cd))


### CI

* **release:** return release-please to the standard flow ([12e93b3](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/12e93b38f88792af44fbdfdc18a6a07c831bee46))

## [0.3.0](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/compare/v0.2.0...v0.3.0) (2026-09-03)


### Features

* **analytics:** a newborn track inherits a just-dead neighbour's staff evidence ([b61e3bf](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/b61e3bff0f9184ee31dd1ca67f09a431d95b296b))
* **assets:** the README figures carry the live dwell field, popless ([cb26eb5](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/cb26eb5978a38def2b618b6ee5fae9e90cdc18e7))
* **assets:** the README figures colour with staff memory ([b2230f9](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/b2230f9c57e93baec9d544e52a47c450feb15636))
* **assets:** the README figures render with person01, and quote its metres ([72a44ae](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/72a44ae78edbc210368f3810caa7ac690c153b54))
* **assets:** the README figures show what each person is doing ([591b352](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/591b352c1b3b8700cd827e5c74773874601b53cd))
* **commissioning:** a dwell/traffic heatmap on the commissioned scene's floor ([00199c0](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/00199c0f0d5f29f2bab554182018169de970b8ab))
* **commissioning:** the demo's figures show what the person is doing ([d2b26e8](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/d2b26e8b992be86e78bca64cf21db60e6515aa37))
* **commissioning:** the dwell field lives on the demo's floor, and figures wear their colour from frame one ([ed0ba6e](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/ed0ba6e770de6dd2c88538ab61933832e496f481))
* **commissioning:** the L1 figure shows what the person is doing ([ff93b20](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/ff93b2001d2e0173c1f69b09411b84f49f62eb4a))
* **geometry:** an equidistant lens model with no pole inside the frame ([c28f362](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/c28f362408069f29c9f45a491d28699f8426d5e7))
* **geometry:** camera.json refuses the resolution and singularity traps ([652ddf4](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/652ddf4ca4907921add09f1de87afb7f18adb6a9))
* **serving:** a uint8 input contract, and the target it was being scored against ([7fa9a94](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/7fa9a94db4d81622a6cd4cc927a4f9967f73f94c))
* **shipped:** name the run once, and make "which checkpoint" a question with a head ([b204593](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/b204593bff38b62ee2150525aafce74654ca217a))
* **shipped:** person01 is the run the tools ship ([9ab0aec](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/9ab0aec7e5f67fd4539be095e098f7a20b22e035))


### Fixes

* **bench:** the end-to-end figure never copied the outputs back ([69ede66](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/69ede6649edce6ca5a5e3cca0fbaa1f2daf07e45))
* **bench:** the printed target said 96x15 beside a constant that is 96x5 ([1cf1121](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/1cf11212d8c95835896a35a20c576d069d61507e))
* **bev3d:** fit_k1 refuses a range that reaches the model's singularity ([9bfe00d](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/9bfe00dfc2233ce5e026a663b1f410f6288e80eb))
* **bev3d:** gate person boxes on the frame the detector saw, not the undistorted one ([4c8566d](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/4c8566da4de7b8087ecf899255645ce902cee2db))
* **commissioning:** heads_video reads the measured constants, not their old values ([322aeee](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/322aeee35d3c78071c563dcb72888aab51278c25))
* **commissioning:** heads_video tests the box centre against the FP polygons ([bfb68ee](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/bfb68ee88c54890d77b2e7ac5a62881e02cc883d))
* **commissioning:** heads_video writes its own record, and can colour staff ([df95b2b](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/df95b2ba53b2dbe2ad9c74bf094ec68234730701))
* **commissioning:** heads_video's figure key partitions the way its colours do ([759c7cd](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/759c7cd6dc2c5a284e33cdf628001f6253f63fe8))
* **data:** the crop branch drops the skeleton with its box ([1452791](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/1452791b5e564ec4cb89d2ddea6f708dbd84f28b))
* **figures:** an exploded skeleton is refused by its own bones ([871f512](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/871f512eaa4a15e8b6baea322f1cc6f6f35aa623))
* **figures:** the staff verdict is passed in, not imported ([e4834d9](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/e4834d9c0ca71f0e01002f7fa15b5c5d0f55aed1))
* **meshes:** a measured limb of zero length killed a 900-frame render ([2c359d2](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/2c359d2a8bd68845ab42251237bca205b2a32c72))
* **meshes:** the head goes on the spine, not toward the nose ([754cd4f](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/754cd4fbe3fb6c41d86a7265d395f6c6a8050b80))
* **meshes:** the posed figure had its head buried in its shoulders ([b67ae84](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/b67ae84ee56727d9a92d8e82a55e9feb476a035b))
* **runmeta:** the selection report could not see a dataset-qualified head ([2387ef6](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/2387ef62777e99c6d25662c24bc95213cef2a39c))
* **scripts:** resplit's refusal says what it does, and does what it says ([8e6ca07](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/8e6ca077cf15ec1e24498b78a353f544ec85f439))
* **scripts:** the campaign's progress log gets its wire ([be8091b](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/be8091bf60ceb4f0375662f1273ddf6d337dee21))
* **site30k:** --calib-root stopped at the first of two readers ([759d4bc](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/759d4bc31f4b6e0274dc3059148cdd900fdfbad7))
* **site30k:** masks_pass could only commission a camera someone else had already built ([d44b0a1](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/d44b0a12e36bf969473f7488cf20566d55933537))


### Performance

* **commissioning:** demo_video renders in parallel, with the track state replayed ([22515ab](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/22515ab4257fd4dc28b9054f857e83a52591be7f))
* **commissioning:** render the frames in parallel, with the track state replayed ([5e107f6](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/5e107f676dad088e0d56b7d3a2623c5ba5a03cdd))
* **meshes:** the shop's normals were recomputed on all 900 frames ([087b088](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/087b0887d5978b75fc19dbd734cb6a85d5fdfd51))


### Refactoring

* **bev3d:** the two functions nothing calls leave the tree ([5c209c7](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/5c209c79bd087679f3e86f0e0cecb7286b9af1e4))
* **cli:** infer_video imports the class-name resolver it was reimplementing ([137b44f](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/137b44fab868307c9a51cbc9b446f832f82c67b6))
* **figures:** the sequential track-state loop has one home ([a79d372](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/a79d372fb92faa7b21eb6c18a07c38058f6733ec))
* **figures:** the shared render code leaves the CLI script it lived in ([e17d0cd](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/e17d0cda74ed81abe19777f2d0a5cfaa446be4bf))
* **models:** the FCOS level flatten has one in-model home ([e346f49](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/e346f49d2788a8bb64b6971cad86b58a128f0dae))
* **scripts:** two hand-rolled normalisations go through the shared ones ([dd8b5a6](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/dd8b5a6bfc2e9a49b281263b02b388c650d13545))


### Documentation

* **contributing:** exports/ is named as the deployment surface it already is ([7195581](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/71955811f2f28069258942f2aac48cdfa17d0adc))
* **figures:** seven render-side comments catch up to the code ([1cad87e](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/1cad87e91a01caff24bb350b460f425836b81a61))
* **hydranet:** sixteen comments catch up to the code they describe ([c36cf0f](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/c36cf0fddc37c4d34d2ce8a9e964ba05f1478931))
* **plan:** person01 answers item 20 -- the site person boxes help every head ([f48527e](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/f48527ea851119517a13b143d1e214d5561d056c))
* **plan:** person01's selection metric was decided in the config and never written back ([083a2d6](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/083a2d698d87d45e89943d9b86bb4fa6156e6a16))
* **plan:** the lift's occlusion failure and its bone gate join 7c.30 ([7161df6](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/7161df6eac52957d7b74ba02040226d9b7e445b3))
* **plan:** the mesh can be driven by the pose head, and the metrics that said it worked could not see it fold ([5ad572b](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/5ad572bed44466b4a827873d308a9e17d49836af))
* **plan:** the serving target is bound by the upload, not the model ([f8181b0](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/f8181b08a04b653503aa2b3668f47ffbd1901bf3))
* **plan:** the stage-0 backlog is 15 selling-floor cameras, counted ([4d4fd12](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/4d4fd12eef2053b1043e2810d5e92ded4dbae884))
* **plan:** the throughput figure was measured against neighbours, not against the card ([ebeadbf](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/ebeadbf9cdad7b8b1f5a6b1d508ecc48e090a634))
* **plan:** the tree's best terrain checkpoint is not the one anything points at ([9022373](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/9022373e6526d422cb5aad85850505b225773b33))
* **readme:** the retail configs train three heads, not two ([1322c00](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/1322c00ff0b96ece3992bb24ab40adcfd97bf2d5))
* **scripts:** four docstrings stop promising what the code does not do ([82e80b7](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/82e80b78171c30d426e4a1580e8999e42707e9ef))
* the dwell field and the popless colours reach the documents ([6a0dd8f](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/6a0dd8fac518063354476f0dc952347fac7d4095))
* three duplicated explanations get one canonical home each ([d89649d](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/d89649d040b9dd394d104814e992ed66b1e969f3))
* **tools:** the README catches up to the twenty tools that exist ([59529b8](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/59529b843c82c899c90aee7d2ee9d60a5d510474))

## [0.2.0](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/compare/v0.1.0...v0.2.0) (2026-08-31)


### Features

* **commissioning:** MapAnything against our own geometry, with the control that makes it mean something ([bac15a2](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/bac15a25486d7bf23597405fb85ebef382b7a8b4))
* **commissioning:** the floor is the only ruler this repo has, so make it one ([b95fa82](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/b95fa824aa7e32aec8e406afc18aed100d2f7dbc))
* **retention:** a product that records customers had no answer to "for how long" ([5ec83c3](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/5ec83c37f7462eb448b2c901e125c5fc002f441e))
* **serving:** the RTSP session policy, without the transport ([6faa621](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/6faa62149aaee824f92d46c87de8a8bdc01c4960))


### Fixes

* **commissioning:** the MapAnything readings were pinned to whatever upstream main said today ([8d9a6f5](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/8d9a6f5de9642c447c39af03f3d4691d20dfd9c1))
* **export:** two ways for simplification not to happen, and neither said so ([2cee1ff](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/2cee1ffcd95f452e3398c6dfd8fa3cf7f8614d0f))
* **packaging:** syncai_bev3d ships annotated and tells consumers it is untyped ([6eb7099](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/6eb709968acf17ea875cd8750fe704f026a08d1d))
* **pose:** a label set's provenance could record an empty string and look filled in ([523e48e](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/523e48e4193d28837c5dfd4c0f17c2aa0d8b6c23))
* **scene:** the renderer that published an unblurred shop floor had no blur stage ([bffa782](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/bffa7824c2fb2bcaa895c673acf1c6ae70e4e3e3))
* **scene:** the third renderer can now be audited, and a relative --out no longer half-writes ([96d011f](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/96d011f9c8cd97faa383c3e28134d91747ea8d00))
* **test:** the commissioned baseline is under runs/, so CI had nothing to read ([27b3f2b](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/27b3f2b7fdb71f4b3b8ecad5aca9a59d841a5c0a))


### Refactoring

* **analytics:** the class that existed only for Python 3.10, and the trap under it ([d21c640](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/d21c640ef17357cd1216a661452afe9d86ba783e))
* **analytics:** three copies of box IoU become one, and a test holds it to the other ([9366957](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/936695770cfb38fbc720c0b16580b179b444b648))
* five more deferred imports, and a reason that stopped being true ([791fda4](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/791fda41be04132be06deeefb4393a09e199e593))
* four deferred imports that deferred nothing, in a package that has been bitten ([16930a8](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/16930a87652d156af2777cab7f025184cbb84dbc))


### Documentation

* **bev3d:** the scene files, and a claim in world.py that had stopped being true ([e59b2db](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/e59b2db7bd97e18f3944b5809f57ba33f5b644fa))
* **bev3d:** two 3D panels, and neither replaced the other ([ef371ea](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/ef371ea2cce9be0f56138ff8c6f738f72498c9db))
* **ci:** the release workflow said two things that were false, under 89 lines of history ([6e1db27](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/6e1db273e01712356ed7590486734aff93dc763e))
* **ci:** the type ratchet was 84% prose, most of it a log of numbers it no longer holds ([0ac2a12](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/0ac2a127f434a574179415834bdde8051ab0aca2))
* four files explained a fix by narrating it; they now state what the code does ([c5eae77](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/c5eae770440953ed711fba195e245fb7288c999b))
* **plan:** the build-order table restated §7 instead of pointing at it ([3032bba](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/3032bba27d709757cce57b7465a5e81cee4581c1))
* three gate files carried a log of their own baselines, and one false claim ([4878f7e](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/4878f7e18c37a16aefa39a58dbc7a7974cb3404a))


### Build & packaging

* Bump transformers from 5.16.0 to 5.16.1 ([1356844](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/1356844f892bf3ffba67eb2aee2c44452e08695e))
* Bump transformers from 5.16.0 to 5.16.1 ([3941d98](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/3941d98d08f9014eefd1ec660b9ac7cb4cdc3ecb))
* Bump ty from 0.0.74 to 0.0.75 in the dev-tooling group ([e6e76c2](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/e6e76c2dafd333ba09350ef64de2e2db1613df6c))
* Bump ty from 0.0.74 to 0.0.75 in the dev-tooling group ([724f119](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/724f1196310152765b9342089914b9442daaf297))
* the Python floor moves to 3.11, and three constraints that existed for 3.10 go ([156727f](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/156727f33caf43e30ca0d6839d2d2109122b45fc))


### CI

* a pull-request check tests a merge that does not exist yet ([b413156](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/b413156ff6c1443e1dc896a21ac350edaee79360))
* release-please comes back, pointed at main ([7a890dd](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/7a890dd3058afe306b83f506bbd4d9209a32713a))
* the coverage floor was two points below what the suite actually holds ([c12059a](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/c12059aae4ce34b196fcc67b0551e1c0ed23c2e4))
* the file was named after a tool that no longer runs here ([e019271](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/e0192717b307c1ca8eae32f74d68f29f0e2be866))
* the hop closest to what ships was the one with no gate ([e116f05](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/e116f0537613fd21620439140263bd160d64a34b))
