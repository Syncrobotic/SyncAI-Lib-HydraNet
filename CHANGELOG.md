# Changelog

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
