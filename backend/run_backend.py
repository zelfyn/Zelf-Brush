"""PyInstaller entry point. `--pick` re-enters here to show a native dialog in a child process."""
import sys

if len(sys.argv) > 1 and sys.argv[1] == "--pick":
    import json

    from rotobrush import dialogs

    a = sys.argv[2:5] + [""] * 3
    print(json.dumps(dialogs.run(a[0], a[1], a[2])))
else:
    from rotobrush.__main__ import main

    main()
