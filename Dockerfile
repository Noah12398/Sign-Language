# Use a base image with Flutter installed
FROM cirrusci/flutter:latest AS build

# Set working directory
WORKDIR /SignLanguage

# Copy only the Flutter project files first (to leverage caching)
COPY pubspec.yaml .
COPY pubspec.lock .
COPY web web
COPY lib lib

# Run pub get
RUN flutter pub get

# Copy the remaining files
COPY . .

# Build the Flutter web project
RUN flutter build web
