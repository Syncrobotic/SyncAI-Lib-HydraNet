# Changelog

## [0.2.0](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/compare/v0.1.0...v0.2.0) (2026-08-16)


### Features

* a second retail taxonomy, for naming objects rather than free space ([31079fa](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/31079facee2187498e16c0c07f47f4b322f213f2))
* accept a directory of stills, for the cameras nobody kept the video of ([c95291f](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/c95291f6e536aa89a350fd82b9d81e9ad987aee6))
* label a fixed camera by consensus, for when nobody can correct a mask ([f0d9e0b](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/f0d9e0bcaf1ee1fd9595b7481126f53aa993adb3))
* let the ROS live view choose the interface it serves on ([af9e7c1](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/af9e7c1763b2e98b7ba39415262c3dc9cefb4999))
* make torch.compile an opt-in training knob ([cf17644](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/cf176448f7d71c8824d03976eb43374ed67746aa))
* people-flow analytics, as post-processing rather than a fourth head ([d594db7](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/d594db7b644784e88ab4774f58c677094cdfd798))
* pre-label with SAM 3 the classes no public dataset supplies ([74dc2d0](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/74dc2d0113e038bb2d2ed8c9c2fa0e4a21a1daca))
* read merchandise out of SAM 3 as instances, not as pixels ([7cd071e](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/7cd071e6a1ecf49907ab728759b456c31d58df3f))


### Fixes

* an instance box the size of a counter is not an instance ([b61ca0c](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/b61ca0c9e0bc3a478a89e1f00733335e509329c3))
* ffprobe refuses the section name every video path asks for ([550138a](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/550138aa1f24124d4577484a839488f25fbdfdbb))
* four annotations that were wrong, not merely unproven ([666a396](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/666a396d67a302809bd642527a5faab8556e4d0f))
* spell the PIL constants the way a static checker can see them ([a3778b0](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/a3778b05232b0fc00613d922c7c892f5ef43d8e5))


### Performance

* batch the EMA update instead of launching a kernel per tensor ([590e35f](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/590e35f8bf25cbbcba46d0a6c2f161eed63eabd1))
* raise workers and channels_last from a measured ceiling, not a guess ([9e4378e](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/9e4378e97119f718adc8eeb191502a688580b5b4))


### Refactoring

* check one dataset entry in one function ([1b4aa8b](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/1b4aa8bac939a52a13ceff253fa9b9f759fba3e4))
* frame selection is library code, not one script importing another ([35dc2f8](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/35dc2f8fb645cad41502ec2ecdf2174ebf26321e))
* give prepare-cocostuff a build_parser, and its first tests ([25f8f1c](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/25f8f1c55c1f74ca292a234630559099721689c4))
* give the preprocessing constants a home below everything ([66cb570](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/66cb570fc54f332d45644a9d85a12ad20627e12a))
* give the two scene payloads a shape a checker can hold you to ([846d920](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/846d9203918fdcfc9721245f91119088b439a6bc))
* lift five concerns out of the Trainer constructor ([89afb98](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/89afb98b3cb02bec102259d4441a407f787e44de))
* make the network iterate its heads instead of unrolling detection ([9a2921d](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/9a2921dcde3fbb3c78ab53de676c54293bc66717))
* move EMA and the optimiser schedule out of the trainer module ([025a34a](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/025a34a7a58da2b216092f023213abcf9a4cdb80))
* move the live view's own work into the package ([c407261](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/c4072617d9b8c458b92468619fc747ac35b369fe))
* narrow the Optionals where the invariant actually holds ([0c2f39f](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/0c2f39fcde777fa2fcb6401cae0236357dac2437))
* read the detection head through one accessor that checks it ([9ee49a1](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/9ee49a186e986efb148b4dd9994bf4bcff7a6c06))
* seed the BiFPN passes from their input, not from None ([2d8e336](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/2d8e3368622f44c84bc1c667bc8a5e0d919f1a47))
* split check_split by the four questions it asks ([63f1b4a](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/63f1b4a607387a336164d5c9df0497af8e2c01ad))
* split evaluate into accumulate and reduce, and fix two annotations ([abac49f](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/abac49f85f34aa0e5d2a709e382dcf30d29c4955))
* split prepare-ade20k's main into clear, link and report ([8755ad6](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/8755ad6ecc708f8f06b517f0503ccf2309bf80ea))
* split the 3D panel renderer along the seams it already had ([ed4eb51](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/ed4eb51e537520bdee132495a333a57dfda8890d))
* thin the scene and infer-video mains, and test what came out ([4b60381](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/4b60381ca8baa1af6637a012f01b3f46d65d2109))


### Documentation

* fit the lens before the pose, and prefer tiles to people ([d08db25](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/d08db25a4192f1cf64c29d751cbf06a07608be5e))
* say what the split hash is for, in the argument that says it ([1a5b56f](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/1a5b56f606437443fb389df7eef4b180e9167c39))


### Build & packaging

* configure pyright, so Pylance stops burying findings in import noise ([2940e6e](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/2940e6ecca41926a6c311c062868e63e661aa2bf))
* keep customer site drawings out of git ([8c7eaa2](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/8c7eaa22db6888e99353dab25b3225766825a81b))
* make a `# noqa` prove it still suppresses something ([e3134e7](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/e3134e79efa409b23e64425ae426266549fbe3a4))
* repair the ty config so the checker starts, and resync the lock ([3b2d12b](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/3b2d12bd82e405a14dba3e761a1ea0333229d12b))
* ship type annotations and wire a checker to read them ([136cd0b](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/136cd0b51e2bd6768e2a74184581ce22addee4be))
* watch the pinned actions and dependencies with dependabot ([3737f75](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/3737f75f6d448a640cd1d8c632f50d567242082d))


### CI

* cut releases with release-please, split across dev and main ([4db1b54](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/4db1b548e90789e5d187a55768472e5be6ade39b))
* install from the lockfile, and hold the type count to a ratchet ([d1d7402](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/d1d740270318c85806e9661fdde04c223eb21d47))
* test the Python the maintainer actually runs ([fbd4b3c](https://github.com/Syncrobotic/SyncAI-Lib-HydraNet/commit/fbd4b3c1bcbe7635458c4cb68167ada23c0cbbad))
