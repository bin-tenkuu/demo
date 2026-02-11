package demo.starter.controller;

import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.servlet.ServletOutputStream;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.http.HttpHeaders;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;
import java.util.zip.GZIPOutputStream;

/**
 * @author bin
 * @since 2025/04/22
 */
@Tag(name = "boom")
@RequestMapping("/boom")
@RestController
public class BoomController {
    private static final int size = 1024 * 8;
    private static final byte[] data = new byte[size];

    @RequestMapping(
            name = "/a.html",
            method = RequestMethod.GET
    )
    public void boom(HttpServletRequest req, HttpServletResponse res) throws IOException {
        var headers = req.getHeaders(HttpHeaders.CONTENT_ENCODING);
        while (headers.hasMoreElements()) {
            var encoding = headers.nextElement();
            System.out.println(encoding);
        }
        ServletOutputStream out;
        try {
            out = res.getOutputStream();
            res.setHeader(HttpHeaders.CONTENT_ENCODING, "gzip");
        } catch (IOException e) {
            return;
        }
        try (var stream = new GZIPOutputStream(out)) {
            for (int i = 0; i < size; i++) {
                stream.write(data, 0, size);
            }
        }
    }
}
