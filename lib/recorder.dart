import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:http/http.dart' as http;
import 'package:record/record.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;
import 'dart:io' show File, Directory;
import 'package:audioplayers/audioplayers.dart';

// فقط للويب
// ignore: avoid_web_libraries_in_flutter
import 'dart:html' as html;

class RecorderPage extends StatefulWidget {
  final String backendBaseUrl;
  RecorderPage({Key? key, this.backendBaseUrl = "http://56.228.70.224:8000"})
      : super(key: key);

  @override
  _RecorderPageState createState() => _RecorderPageState();
}

class _RecorderPageState extends State<RecorderPage> {
  List<PlatformFile> selectedFiles = [];
  List<String> uploadedPaths = [];
  TextEditingController textController = TextEditingController();

  String status = '';
  String? lastDownloadUrl;
  bool uploading = false;

  bool useDiacritization = true;
  String voiceMode = "default"; // default, clone, merge

  // تسجيل الصوت على الموبايل
  bool isRecording = false;
  Record record = Record();
  String? currentFilePath;

  // تسجيل الصوت على الويب
  html.MediaRecorder? webRecorder;
  List<html.Blob> webChunks = [];

  // لتشغيل الصوت
  AudioPlayer? audioPlayer;

  @override
  void dispose() {
    textController.dispose();
    audioPlayer?.dispose();
    super.dispose();
  }

  /// اختيار ملفات من الجهاز
  Future<void> pickFiles() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['wav', 'mp3', 'm4a', 'ogg', 'flac'],
      allowMultiple: true,
      withData: true,
    );

    if (result != null) {
      setState(() => selectedFiles.addAll(result.files));
    }
  }

  /// رفع جميع الملفات للباكند
  Future<void> uploadAll() async {
    if (selectedFiles.isEmpty) {
      setState(() => status = "لا توجد ملفات للرفع");
      return;
    }

    setState(() {
      uploading = true;
      status = "Uploading...";
      uploadedPaths = [];
    });

    try {
      var uri = Uri.parse("${widget.backendBaseUrl}/upload_speakers");
      var req = http.MultipartRequest('POST', uri);

      for (var f in selectedFiles) {
        if (kIsWeb) {
          if (f.bytes == null) continue;
          req.files.add(http.MultipartFile.fromBytes(
              'files', f.bytes!, filename: f.name));
        } else {
          if (f.path == null) continue;
          req.files.add(
              await http.MultipartFile.fromPath('files', f.path!, filename: f.name));
        }
      }

      var streamed = await req.send();
      var res = await http.Response.fromStream(streamed);

      var j = jsonDecode(res.body);
      if (j["ok"] == true) {
        setState(() {
          uploadedPaths = List<String>.from(j["paths"]);
          status = "Uploaded ${uploadedPaths.length} files.";
        });
      } else {
        setState(() => status = "Upload failed: ${res.body}");
      }
    } catch (e) {
      setState(() => status = "Upload Error: $e");
    }

    setState(() => uploading = false);
  }

  /// توليد الصوت من النص وتشغيله مباشرة
  Future<void> synthesize() async {
    final text = textController.text.trim();

    if (text.isEmpty) {
      setState(() => status = "اكتب النص اولاً");
      return;
    }

    setState(() {
      uploading = true;
      status = "Synthesis started…";
      lastDownloadUrl = null;
    });

    try {
      var uri = Uri.parse("${widget.backendBaseUrl}/synthesize");

      var body = {
        "text": text,
        "use_diacritization": useDiacritization ? "true" : "false",
        "mode": voiceMode,
        "speaker_paths": uploadedPaths.join(",")
      };

      var res = await http.post(uri, body: body);
      var j = jsonDecode(res.body);

      if (j["audio_url"] != null) {
        setState(() {
          lastDownloadUrl = widget.backendBaseUrl + j["audio_url"];
          status = j["log"] ?? "Done";
        });

        // تشغيل الصوت مباشرة
        audioPlayer ??= AudioPlayer();
        await audioPlayer!.stop();
        await audioPlayer!.play(UrlSource(lastDownloadUrl!));
      } else {
        setState(() => status = "Error: ${res.body}");
      }
    } catch (e) {
      setState(() => status = "Synthesis error: $e");
    }

    setState(() => uploading = false);
  }

  /// تسجيل صوتي على الموبايل / Emulator أو Web
  Future<void> toggleRecording() async {
    if (kIsWeb) {
      if (webRecorder != null && webRecorder!.state == "recording") {
        webRecorder!.stop();
        setState(() => status = "Processing Web recording...");
      } else {
        webChunks = [];
        var stream = await html.window.navigator.mediaDevices!
            .getUserMedia({'audio': true});
        webRecorder = html.MediaRecorder(stream);
        webRecorder!.addEventListener('dataavailable', (event) {
          final e = event as html.BlobEvent;
          webChunks.add(e.data!);
        });
        webRecorder!.addEventListener('stop', (event) async {
          final blob = html.Blob(webChunks);
          final reader = html.FileReader();
          reader.readAsArrayBuffer(blob);
          reader.onLoadEnd.listen((_) {
            final bytes = reader.result as Uint8List;
            selectedFiles.add(PlatformFile(
              name: 'web_record_${DateTime.now().millisecondsSinceEpoch}.webm',
              bytes: bytes,
              size: bytes.length,
            ));
            setState(() => status = "تم تسجيل الصوت على Web");
          });
        });
        webRecorder!.start();
        setState(() => status = "Recording on Web...");
      }
      return;
    }

    if (isRecording) {
      final path = await record.stop();
      setState(() => isRecording = false);

      if (path != null) {
        final file = File(path);
        selectedFiles.add(PlatformFile(
          name: p.basename(path),
          path: path,
          size: file.lengthSync(),
        ));
        setState(() => status = "تم تسجيل الصوت: ${p.basename(path)}");
      }
    } else {
      if (await record.hasPermission()) {
        Directory tempDir = await getTemporaryDirectory();
        final filePath =
        p.join(tempDir.path, 'voice_${DateTime.now().millisecondsSinceEpoch}.m4a');
        currentFilePath = filePath;

        await record.start(
          path: filePath,
          encoder: AudioEncoder.aacLc,
          bitRate: 128000,
          samplingRate: 44100,
        );

        setState(() {
          isRecording = true;
          status = "Recording...";
        });
      } else {
        setState(() => status = "لا يوجد إذن لتسجيل الصوت");
      }
    }
  }

  Widget fileTile(PlatformFile f) {
    return ListTile(
      title: Text(f.name),
      subtitle: Text("${(f.size / 1024).toStringAsFixed(1)} KB"),
      trailing: IconButton(
        icon: Icon(Icons.delete),
        onPressed: () => setState(() => selectedFiles.remove(f)),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("XTTS Client")),
      body: Padding(
        padding: EdgeInsets.all(12),
        child: Column(
          children: [
            DropdownButtonFormField(
              value: voiceMode,
              items: [
                DropdownMenuItem(value: "default", child: Text("Default Voice")),
                DropdownMenuItem(value: "clone", child: Text("Voice Cloning")),
                DropdownMenuItem(value: "merge", child: Text("Merge Voices")),
              ],
              onChanged: (v) => setState(() => voiceMode = v.toString()),
              decoration: InputDecoration(labelText: "وضع الصوت"),
            ),

            SwitchListTile(
              title: Text("تفعيل التشكيل التلقائي"),
              value: useDiacritization,
              onChanged: (v) => setState(() => useDiacritization = v),
            ),

            Row(
              children: [
                ElevatedButton.icon(
                  icon: Icon(isRecording ? Icons.stop : Icons.mic),
                  label: Text(isRecording ? "Stop Recording" : "Record Voice"),
                  onPressed: toggleRecording,
                ),
                SizedBox(width: 8),
                ElevatedButton.icon(
                  icon: Icon(Icons.attach_file),
                  label: Text("اختر ملفات الصوت"),
                  onPressed: pickFiles,
                ),
              ],
            ),

            Expanded(
              child: selectedFiles.isEmpty
                  ? Center(child: Text("لا توجد ملفات مختارة"))
                  : ListView(children: selectedFiles.map(fileTile).toList()),
            ),

            TextField(
              controller: textController,
              maxLines: 3,
              decoration: InputDecoration(
                labelText: "النص",
                border: OutlineInputBorder(),
              ),
            ),

            Row(
              children: [
                Expanded(
                  child: ElevatedButton(
                    onPressed: uploading ? null : uploadAll,
                    child: Text("Upload"),
                  ),
                ),
                SizedBox(width: 8),
                Expanded(
                  child: ElevatedButton(
                    onPressed: uploading ? null : synthesize,
                    child: Text("Synthesize"),
                  ),
                ),
              ],
            ),

            SizedBox(height: 8),
            SelectableText("Status: $status"),

            if (lastDownloadUrl != null)
              Row(
                children: [
                  ElevatedButton(
                    onPressed: () async {
                      audioPlayer ??= AudioPlayer();
                      await audioPlayer!.play(UrlSource(lastDownloadUrl!));
                    },
                    child: Text("Play"),
                  ),
                  SizedBox(width: 8),
                  ElevatedButton(
                    onPressed: () async {
                      if (audioPlayer != null) await audioPlayer!.stop();
                    },
                    child: Text("Stop"),
                  ),
                ],
              ),
          ],
        ),
      ),
    );
  }
}
