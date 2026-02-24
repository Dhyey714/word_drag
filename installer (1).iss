[Setup]
AppName=WordDrag
AppVersion=2.0
DefaultDirName={autopf}\WordDrag
DefaultGroupName=WordDrag
OutputBaseFilename=WordDrag_Setup
Compression=lzma
SolidCompression=yes

[Files]
Source: "dist\word_drag\*"; DestDir: "{app}"; Flags: recursesubdirs

[Icons]
Name: "{group}\WordDrag"; Filename: "{app}\WordDrag.exe"
Name: "{commondesktop}\WordDrag"; Filename: "{app}\WordDrag.exe"

[Run]
Filename: "{app}\WordDrag.exe"; Description: "Launch WordDrag"; Flags: postinstall
