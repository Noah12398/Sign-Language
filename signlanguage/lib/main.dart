import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:http/http.dart' as http;
import 'package:camera/camera.dart';
import 'package:image_picker/image_picker.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Get available cameras
  List<CameraDescription> cameras = [];
  try {
    cameras = await availableCameras();
  } catch (e) {
    print('Error initializing cameras: $e');
  }

  runApp(MyApp(cameras: cameras));
}

class MyApp extends StatelessWidget {
  final List<CameraDescription> cameras;

  const MyApp({Key? key, required this.cameras}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Sign Language Translator',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        brightness: Brightness.light,
      ),
      darkTheme: ThemeData(
        primarySwatch: Colors.blue,
        brightness: Brightness.dark,
      ),
      themeMode: ThemeMode.system,
      home: MyHomePage(cameras: cameras),
      debugShowCheckedModeBanner: false,
    );
  }
}

class MyHomePage extends StatefulWidget {
  final List<CameraDescription> cameras;

  const MyHomePage({Key? key, required this.cameras}) : super(key: key);

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> with WidgetsBindingObserver {
  CameraController? _controller;
  Future<void>? _initializeControllerFuture;
  late FlutterTts _flutterTts;
  final ImagePicker _picker = ImagePicker();

  String _recognizedText = "No sign detected";
  String _sentence = "";
  List<String> _letterBuffer = [];
  bool isProcessing = false;
  bool isSpeaking = false;
  String _statusMessage = "Ready to detect signs";
  bool _isError = false;
  int _detectionAttempts = 0;
  Timer? _continuousDetectionTimer;
  bool _continuousDetection = false;
  bool _cameraAvailable = false;

  // Server details - change to your Python server address
  final String serverUrl =
      'https://sign-language-sign-language.up.railway.app'; // Default for Android emulator

  @override
  void initState() {
    super.initState();

    // Add observer for app lifecycle changes
    WidgetsBinding.instance.addObserver(this);

    // Initialize text-to-speech
    _flutterTts = FlutterTts();
    _flutterTts.setLanguage('en-US');

    // Don't initialize the camera here, do it in onNewCameraSelected
    if (widget.cameras.isNotEmpty) {
      onNewCameraSelected(widget.cameras[0]);
    } else {
      setState(() {
        _cameraAvailable = false;
        _statusMessage = "No camera available. Use gallery images instead.";
      });
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // Handle app lifecycle state changes
    final CameraController? cameraController = _controller;

    // App state changed before we got the chance to initialize the camera
    if (cameraController == null || !cameraController.value.isInitialized) {
      return;
    }

    if (state == AppLifecycleState.inactive) {
      // Free up memory when the app is inactive and camera isn't needed
      cameraController.dispose();
    } else if (state == AppLifecycleState.resumed) {
      // Reinitialize camera with the same camera
      onNewCameraSelected(cameraController.description);
    }
  }

  void onNewCameraSelected(CameraDescription cameraDescription) async {
    // Dispose the previous controller
    if (_controller != null) {
      await _controller!.dispose();
    }

    // Create a new controller
    final CameraController controller = CameraController(
      cameraDescription,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );

    _controller = controller;

    // If the controller is updated then update the UI.
    controller.addListener(() {
      if (mounted) setState(() {});
    });

    try {
      _initializeControllerFuture = controller.initialize();
      await _initializeControllerFuture;
      setState(() {
        _cameraAvailable = true;
        _statusMessage = "Camera initialized. Ready to detect signs.";
      });
    } catch (e) {
      setState(() {
        _cameraAvailable = false;
        _statusMessage =
            "Camera error: ${e.toString().split('\n')[0]}. Use gallery images instead.";
        _isError = true;
      });
      print('Error initializing camera: $e');
    }

    if (mounted) {
      setState(() {});
    }
  }

  @override
  void dispose() {
    // Remove observer
    WidgetsBinding.instance.removeObserver(this);

    // Dispose the controller
    _controller?.dispose();
    _flutterTts.stop();
    _continuousDetectionTimer?.cancel();
    super.dispose();
  }

  Future<void> _takePictureAndDetect() async {
    if (!_cameraAvailable || _controller == null) {
      _showCameraErrorDialog();
      return;
    }

    setState(() {
      isProcessing = true;
      _statusMessage = "Processing image...";
      _isError = false;
      _detectionAttempts++;
    });

    try {
      // Ensure camera is initialized
      await _initializeControllerFuture;

      // Take the picture
      final XFile image = await _controller!.takePicture();

      // Process the image
      await _processImageFile(image);
    } catch (e) {
      setState(() {
        _recognizedText = "Capture failed";
        _statusMessage =
            "Error capturing image: ${e.toString().split('\n')[0]}";
        _isError = true;
      });
      print('Error taking picture: $e');
    } finally {
      setState(() {
        isProcessing = false;
      });

      // Continue detection if continuous mode is enabled
      // Continue detection if continuous mode is enabled
      if (_continuousDetection && !isProcessing) {
        _continuousDetectionTimer = Timer(Duration(milliseconds: 5000), () {
          if (mounted && _continuousDetection) {
            _takePictureAndDetect();
          }
        });
      }
    }
  }

  // Rest of your code remains the same...

  Future<void> _pickImageFromGallery() async {
    try {
      final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
      if (image != null) {
        setState(() {
          isProcessing = true;
          _statusMessage = "Processing gallery image...";
          _isError = false;
          _detectionAttempts++;
        });

        await _processImageFile(image);
      }
    } catch (e) {
      setState(() {
        _statusMessage = "Error picking image: ${e.toString().split('\n')[0]}";
        _isError = true;
      });
    } finally {
      setState(() {
        isProcessing = false;
      });
    }
  }

  Future<void> _processImageFile(XFile imageFile) async {
    try {
      // Read image file
      final bytes = await imageFile.readAsBytes();
      final base64Image = base64Encode(bytes);

      // Send to server for processing
      final response = await http
          .post(
        Uri.parse('$serverUrl/detect_sign_from_image'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'image': base64Image}),
      )
          .timeout(
        const Duration(seconds: 20),
        onTimeout: () {
          throw TimeoutException('Server is taking too long to respond');
        },
      );

      if (response.statusCode == 200) {
        final jsonData = json.decode(response.body);
        final detectedChar = jsonData['character'] ?? "?";

        setState(() {
          _recognizedText = detectedChar;
          _letterBuffer.add(detectedChar);
          _statusMessage = "Sign detected successfully!";
          _isError = false;

          // Process letter buffer to form words/sentences
          _processLetterBuffer();
        });
      } else if (response.statusCode == 400) {
        setState(() {
          _recognizedText = "No sign detected";
          _statusMessage =
              "No hand detected. Please ensure your hand is clearly visible.";
          _isError = true;
        });
      } else {
        setState(() {
          _recognizedText = "Error occurred";
          _statusMessage = "Server error: ${response.statusCode}";
          _isError = true;
        });
      }
    } catch (e) {
      setState(() {
        _recognizedText = "Connection failed";
        _statusMessage =
            "Error connecting to server: ${e.toString().split('\n')[0]}";
        _isError = true;
      });
    }
  }

  void _processLetterBuffer() {
    // Simple logic to detect spaces and form words
    if (_letterBuffer.isNotEmpty) {
      // Check for special gestures or repeated letters
      final lastLetter = _letterBuffer.last;

      // Example: If S appears twice in sequence, add a space
      if (_letterBuffer.length >= 2 &&
          _letterBuffer[_letterBuffer.length - 1] == 'S' &&
          _letterBuffer[_letterBuffer.length - 2] == 'S') {
        _sentence += " ";
        _letterBuffer.clear();
        return;
      }

      // Example: If user holds the same sign for 3+ frames, consider it deliberate
      if (_letterBuffer.length >= 2 &&
          _letterBuffer
              .sublist(_letterBuffer.length - 3)
              .every((e) => e == lastLetter)) {
        _sentence += lastLetter;
        _letterBuffer.clear();
      }

      // Limit buffer size to prevent memory issues
      if (_letterBuffer.length > 20) {
        _letterBuffer.removeAt(0);
      }
    }
  }

  void _toggleContinuousDetection() {
    if (!_cameraAvailable) {
      _showCameraErrorDialog();
      return;
    }

    setState(() {
      _continuousDetection = !_continuousDetection;
      if (_continuousDetection) {
        _statusMessage = "Continuous detection enabled";
        _takePictureAndDetect();
      } else {
        _statusMessage = "Continuous detection disabled";
        _continuousDetectionTimer?.cancel();
      }
    });
  }

  Future<void> _speakText(String text) async {
    if (text.isNotEmpty && !isSpeaking) {
      setState(() {
        isSpeaking = true;
      });
      await _flutterTts.speak(text);
      setState(() {
        isSpeaking = false;
      });
    }
  }

  void _clearSentence() {
    setState(() {
      _sentence = "";
      _letterBuffer.clear();
      _statusMessage = "Sentence cleared";
      _isError = false;
    });
  }

  void _showCameraErrorDialog() {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: Text("Camera Unavailable"),
          content: Text(
              "The camera is not available or could not be initialized. "
              "You can still use the app by selecting images from your gallery."),
          actions: [
            TextButton(
              child: Text("Use Gallery"),
              onPressed: () {
                Navigator.of(context).pop();
                _pickImageFromGallery();
              },
            ),
            TextButton(
              child: Text("Close"),
              onPressed: () {
                Navigator.of(context).pop();
              },
            ),
          ],
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Sign Language Translator'),
        actions: [
          IconButton(
            icon: Icon(Icons.photo),
            onPressed: _pickImageFromGallery,
            tooltip: 'Choose from gallery',
          ),
          if (_cameraAvailable)
            IconButton(
              icon: Icon(_continuousDetection ? Icons.pause : Icons.play_arrow),
              onPressed: _toggleContinuousDetection,
              tooltip: _continuousDetection
                  ? 'Pause detection'
                  : 'Continuous detection',
            ),
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _clearSentence,
            tooltip: 'Clear sentence',
          ),
        ],
      ),
      body: Column(
        children: [
          // Camera preview or placeholder
          Expanded(
            flex: 3,
            child: _cameraAvailable && _controller != null
                ? FutureBuilder<void>(
                    future: _initializeControllerFuture,
                    builder: (context, snapshot) {
                      if (snapshot.connectionState == ConnectionState.done) {
                        return Stack(
                          alignment: Alignment.center,
                          children: [
                            CameraPreview(_controller!),
                            if (isProcessing)
                              Container(
                                color: Colors.black.withOpacity(0.3),
                                child: Center(
                                  child: CircularProgressIndicator(),
                                ),
                              ),
                            // Hand placement guide
                            Positioned(
                              top: 0,
                              left: 0,
                              right: 0,
                              bottom: 0,
                              child: Container(
                                decoration: BoxDecoration(
                                  border: Border.all(
                                    color: Colors.white.withOpacity(0.5),
                                    width: 2,
                                  ),
                                  borderRadius: BorderRadius.circular(20),
                                ),
                                margin: EdgeInsets.all(50),
                              ),
                            ),
                            // Guidance text
                            Positioned(
                              bottom: 10,
                              child: Container(
                                padding: EdgeInsets.symmetric(
                                    horizontal: 16, vertical: 8),
                                decoration: BoxDecoration(
                                  color: Colors.black.withOpacity(0.5),
                                  borderRadius: BorderRadius.circular(20),
                                ),
                                child: Text(
                                  "Place your hand in the center",
                                  style: TextStyle(color: Colors.white),
                                ),
                              ),
                            ),
                          ],
                        );
                      } else {
                        return Center(child: CircularProgressIndicator());
                      }
                    },
                  )
                : Container(
                    color: Colors.grey.shade900,
                    child: Center(
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(
                            Icons.camera_alt_outlined,
                            size: 80,
                            color: Colors.grey.shade400,
                          ),
                          SizedBox(height: 16),
                          Text(
                            "Camera not available",
                            style: TextStyle(
                              fontSize: 18,
                              color: Colors.grey.shade400,
                            ),
                          ),
                          SizedBox(height: 24),
                          ElevatedButton.icon(
                            icon: Icon(Icons.photo),
                            label: Text("Select from Gallery"),
                            onPressed: _pickImageFromGallery,
                          ),
                        ],
                      ),
                    ),
                  ),
          ),

          // Status message
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(10),
            margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            decoration: BoxDecoration(
              color: _isError
                  ? Colors.red.withOpacity(0.1)
                  : Colors.blue.withOpacity(0.1),
              borderRadius: BorderRadius.circular(8),
              border: Border.all(
                color: _isError ? Colors.red : Colors.blue,
              ),
            ),
            child: Row(
              children: [
                Icon(
                  _isError ? Icons.error_outline : Icons.info_outline,
                  color: _isError ? Colors.red : Colors.blue,
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(
                    _statusMessage,
                    style: TextStyle(
                      color: _isError ? Colors.red : null,
                    ),
                  ),
                ),
              ],
            ),
          ),

          // Detected sign
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  'Current Sign: ',
                  style: const TextStyle(
                      fontSize: 18, fontWeight: FontWeight.bold),
                ),
                Container(
                  width: 60,
                  height: 60,
                  decoration: BoxDecoration(
                    color: Theme.of(context).colorScheme.primaryContainer,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Center(
                    child: Text(
                      _recognizedText,
                      style: TextStyle(
                        fontSize: 24,
                        fontWeight: FontWeight.bold,
                        color: Theme.of(context).colorScheme.onPrimaryContainer,
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),

          // Translated sentence
          Expanded(
            flex: 2,
            child: Container(
              margin: const EdgeInsets.all(16),
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Theme.of(context).cardColor,
                borderRadius: BorderRadius.circular(8),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.1),
                    blurRadius: 4,
                    offset: const Offset(0, 2),
                  ),
                ],
              ),
              width: double.infinity,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Text(
                        'Translated Sentence:',
                        style: TextStyle(
                            fontSize: 18, fontWeight: FontWeight.bold),
                      ),
                      Row(
                        children: [
                          if (_sentence.isNotEmpty)
                            IconButton(
                              icon: Icon(
                                isSpeaking ? Icons.volume_off : Icons.volume_up,
                                color: isSpeaking ? Colors.blue : null,
                              ),
                              onPressed: () => _speakText(_sentence),
                              tooltip: 'Speak sentence',
                            ),
                        ],
                      ),
                    ],
                  ),
                  const SizedBox(height: 10),
                  Expanded(
                    child: SingleChildScrollView(
                      child: Text(
                        _sentence.isEmpty
                            ? 'No signs translated yet'
                            : _sentence,
                        style:
                            const TextStyle(fontSize: 22, letterSpacing: 1.2),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),

          // Manual detection button (only shown if camera is available and continuous mode is off)
          if (_cameraAvailable && !_continuousDetection)
            Padding(
              padding: const EdgeInsets.only(bottom: 20),
              child: ElevatedButton.icon(
                onPressed: isProcessing ? null : _takePictureAndDetect,
                icon: const Icon(Icons.sign_language, size: 32),
                label: const Text(
                  "Detect Sign",
                  style: TextStyle(fontSize: 20),
                ),
                style: ElevatedButton.styleFrom(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 50, vertical: 16),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(10),
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }
}
