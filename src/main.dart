import 'dart:async';
import 'dart:convert';
import 'dart:io';

void main() async {
  IntentClassifier ic = new IntentClassifier();
  await ic.init();
  while (true) {
    String input = stdin.readLineSync();
    String result = await ic.parseIntent(input);
    print(result);
  }
}

class IntentClassifier {
  Process proc;
  Completer task;

  Future<void> init() async {
    proc = await Process.start(
        "D:\\PersonalProjects\\iris-core-nlu\\Scripts\\python.exe",
        ["D:\\PersonalProjects\\iris-core-nlu\\src\\Model.py", "start"]);
    ;

    proc.stdout.transform(utf8.decoder).listen((data) {
      task?.complete(data.trim());
    });
  }

  Future<String> parseIntent(String text) {
    task = new Completer<String>();
    proc.stdin.writeln(text);

    return task.future;
  }
}
