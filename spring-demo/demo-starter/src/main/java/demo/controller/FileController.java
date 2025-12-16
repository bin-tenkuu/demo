package demo.controller;

import demo.model.ResultModel;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.jetbrains.annotations.NotNull;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;
import org.springframework.http.*;
import org.springframework.lang.Nullable;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.*;
import java.net.URI;
import java.net.URL;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.Objects;

/**
 * @author bin
 * @since 2025/04/30
 */
@SuppressWarnings("ResultOfMethodCallIgnored")
@Tag(name = "file")
@Slf4j
@RestController
@RequestMapping("/file")
@RequiredArgsConstructor
public class FileController {

    private final File uploadDir = new File("/home/bin-/code/cms-huabei/target");

    @PostMapping("/upload")
    public ResultModel<String> upload(@RequestParam(required = false) String name, @RequestPart MultipartFile file)
            throws IOException {
        uploadDir.mkdirs();
        if (name == null) {
            name = file.getOriginalFilename();
            if (name == null) {
                throw new IllegalArgumentException("文件名不能为空");
            }
        }
        var filePath = new File(uploadDir, name);
        file.transferTo(filePath.toPath());
        return ResultModel.success("/download/path/" + name);
    }

    @GetMapping("/download/{url}/**")
    public ResponseEntity<Resource> download(
            HttpServletRequest request,
            @RequestHeader(value = HttpHeaders.RANGE, required = false) String rangeStr,
            @PathVariable("url") String url
    ) throws IOException {
        var fullUrl = request.getRequestURI();
        var fileName = fullUrl.substring(fullUrl.indexOf("/download/" + url) - 1 + "/download/".length());
        var filePath = new File(uploadDir, fileName);
        if (!filePath.exists() || !filePath.canRead()) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).build();
        }
        var contentLength = filePath.length();
        var encodedFileName = encodedFileName(fileName);

        var httpRanges = HttpRange.parseRanges(rangeStr);
        if (httpRanges.isEmpty()) {
            // 完整文件下载
            return ResponseEntity.ok()
                    .contentType(MediaType.APPLICATION_OCTET_STREAM)
                    .contentLength(filePath.length())
                    .header(HttpHeaders.CONTENT_DISPOSITION, encodedFileName)
                    .header(HttpHeaders.ACCEPT_RANGES, "bytes")
                    .body(new FileSystemResource(filePath));
        }
        var range = httpRanges.getFirst();
        long rangeStart = range.getRangeStart(contentLength);
        long rangeEnd = range.getRangeEnd(contentLength);
        long rangeLength = Math.min(rangeEnd - rangeStart + 1, contentLength - rangeStart);

        // 创建资源区域
        return ResponseEntity.status(HttpStatus.PARTIAL_CONTENT)
                .contentType(MediaType.APPLICATION_OCTET_STREAM)
                .contentLength(rangeLength)
                .header(HttpHeaders.CONTENT_DISPOSITION, encodedFileName)
                .header(HttpHeaders.ACCEPT_RANGES, "bytes")
                .header(HttpHeaders.CONTENT_RANGE, "bytes " + rangeStart + "-" +
                                                   (rangeStart + rangeLength - 1) + "/" + contentLength)
                .body(new RegionFileResource(filePath, rangeStart, rangeLength));
    }

    private static String encodedFileName(String url) {
        return "attachment;filename=" + URLEncoder.encode(Objects.requireNonNull(url), StandardCharsets.UTF_8);
    }

    private final static class RegionFileResource extends InputStream implements Resource {
        @NotNull
        private final File file;
        private final RandomAccessFile raf;
        private final long position;
        private final long length;
        private long remainingLength;

        public RegionFileResource(@NotNull File file, long position, long length) throws IOException {
            this.file = file;
            this.raf = new RandomAccessFile(file, "r");
            raf.seek(position);
            this.position = position;
            this.length = length;
            this.remainingLength = length;
        }

        @NotNull
        @Override
        public InputStream getInputStream() {
            return this;
        }

        @Override
        public long contentLength() {
            return length;
        }

        @NotNull
        @Override
        public String getDescription() {
            return "file [" + file.getAbsolutePath() + "]" + "[" + position + "+" + (position + length) + "]";
        }

        @Override
        public boolean exists() {
            return file.exists();
        }

        @Override
        public boolean isReadable() {
            return file.canRead() && !file.isDirectory();
        }

        @Override
        public @NotNull URL getURL() throws IOException {
            return file.toURI().toURL();
        }

        @Override
        public @NotNull URI getURI() {
            return file.toURI();
        }

        @Override
        public boolean isFile() {
            return true;
        }

        @Override
        public @NotNull File getFile() {
            return file;
        }

        @Override
        public long lastModified() {
            return file.lastModified();
        }

        @Override
        public @NotNull Resource createRelative(@NotNull String relativePath) throws IOException {
            throw new FileNotFoundException("Cannot create a relative resource for " + getDescription());
        }

        @Override
        public String getFilename() {
            return file.getName();
        }

        @Override
        public boolean equals(@Nullable Object other) {
            return this == other
                   || other instanceof RegionFileResource that
                      && file.equals(that.file)
                      && position == that.position
                      && length == that.length;
        }

        @Override
        public int hashCode() {
            return file.hashCode();
        }

        @Override
        public int read() throws IOException {
            if (remainingLength <= 0) {
                return -1;
            }
            remainingLength--;
            return raf.read();
        }

        @Override
        public int read(byte @NotNull [] b, int off, int len) throws IOException {
            if (remainingLength <= 0) {
                return -1;
            }
            int maxRead = (int) Math.min(len, remainingLength);
            int read = raf.read(b, off, maxRead);
            if (read > 0) {
                remainingLength -= read;
            }
            return read;
        }

        @Override
        public void close() throws IOException {
            raf.close();
        }
    }
}
