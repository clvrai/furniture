# Build Unity binary

For easy deployment on servers, we provide an executable version of Unity instead of having to setup the editor. The code automatically looks for a binary in the path `furniture/binary/Furniture.app` or `furniture/binary/Furniture.x86_64`.

To build your own binary, follow the steps:
1. Open `All` scene in the Unity editor.
2. Go to `File > Build settings`.
3. Check the scene `All` to build. If the scene is not on the list, click `Add Open Scenes`.
4. Set an appropriate target platform (Mac OS X, Windows, or Linux).
5. Set `x86_64` for Architecture.
6. Do **not** check `headless mode` (it will not render anything).
7. Click `Build`.
8. Check whether all files are created.
For macOS, only `Furniture.app` is generated.
For Ubuntu, `Furniture.x86_64` and `Furniture_Data/` are generated.

