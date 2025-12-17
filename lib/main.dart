import 'package:flutter/material.dart';
import 'recorder.dart';

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext ctx) {
    return MaterialApp(
      title: 'XTTS Flutter',
      home: RecorderPage(),
    );
  }
}
