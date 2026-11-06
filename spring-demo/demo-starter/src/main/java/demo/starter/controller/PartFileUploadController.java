package demo.starter.controller;

import demo.core.model.ResultModel;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * @author bin
 * @since 2026/09/24
 */
@Slf4j
@Tag(name = "PartFile")
@RestController
@RequestMapping("/api/PartFile")
@RequiredArgsConstructor
public class PartFileUploadController {
    private static final Path uploadDir;

    static {
        uploadDir = Path.of("./");
        // try {
        //     uploadDir = Files.createTempDirectory("partFile");
        //     log.info("Upload directory created: {}", uploadDir);
        // } catch (IOException e) {
        //     throw new RuntimeException(e);
        // }
    }

    @PostMapping("/upload")
    public void upload(
            @RequestParam String name,
            @RequestParam int currentPart,
            @RequestPart MultipartFile partFile
    ) throws IOException {
        var partPath = uploadDir.resolve(name + "." + currentPart);
        try (var out = Files.newOutputStream(partPath)) {
            partFile.getInputStream().transferTo(out);
        }
    }

    @PostMapping("/uploadComplete")
    public ResultModel<String> uploadComplete(
            @RequestParam String name,
            @RequestParam int totalPart
    ) throws IOException {
        var allPath = uploadDir.resolve(name);
        String failMsg = null;
        try (var out = Files.newOutputStream(allPath)) {
            for (int i = 0; i < totalPart; i++) {
                var partPath = uploadDir.resolve(name + "." + i);
                if (failMsg != null) {
                    Files.deleteIfExists(partPath);
                    continue;
                } else if (!Files.isRegularFile(partPath)) {
                    failMsg = "Part file missing: " + partPath;
                    continue;
                }
                try (var in = Files.newInputStream(partPath)) {
                    in.transferTo(out);
                } catch (IOException e) {
                    failMsg = "Error reading part file: " + partPath;
                }
                Files.deleteIfExists(partPath);
            }
        } catch (IOException e) {
            failMsg = "Error writing file: " + allPath;
        }
        if (failMsg != null) {
            Files.deleteIfExists(allPath);
            return ResultModel.fail(failMsg);
        }
        return ResultModel.success(allPath.toString());
    }

}
