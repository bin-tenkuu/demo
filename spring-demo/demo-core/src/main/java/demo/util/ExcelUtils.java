package demo.util;

import cn.idev.excel.EasyExcel;
import cn.idev.excel.ExcelReader;
import cn.idev.excel.ExcelWriter;
import cn.idev.excel.context.AnalysisContext;
import cn.idev.excel.read.listener.ReadListener;
import cn.idev.excel.read.metadata.ReadSheet;
import cn.idev.excel.support.ExcelTypeEnum;
import cn.idev.excel.write.handler.SheetWriteHandler;
import cn.idev.excel.write.metadata.WriteSheet;
import cn.idev.excel.write.metadata.style.WriteCellStyle;
import cn.idev.excel.write.style.HorizontalCellStyleStrategy;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.FormulaEvaluator;
import org.apache.poi.ss.usermodel.HorizontalAlignment;
import org.apache.poi.ss.usermodel.Workbook;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.ResponseEntity;

import java.io.*;
import java.net.URLEncoder;
import java.nio.ByteBuffer;
import java.nio.channels.ReadableByteChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Collections;
import java.util.List;

@Slf4j
@SuppressWarnings("unused")
public class ExcelUtils {

    public static HorizontalCellStyleStrategy getWriteHandler() {
        // 表头样式
        WriteCellStyle headWriteCellStyle = new WriteCellStyle();
        // 设置表头居中对齐
        headWriteCellStyle.setHorizontalAlignment(HorizontalAlignment.CENTER);
        // 内容样式
        WriteCellStyle contentWriteCellStyle = new WriteCellStyle();
        // 设置内容靠左对齐
        contentWriteCellStyle.setHorizontalAlignment(HorizontalAlignment.LEFT);
        return new HorizontalCellStyleStrategy(headWriteCellStyle, contentWriteCellStyle);
    }

    /**
     * 读取excel文件
     */
    public static ExcelReader readExcel(InputStream ins) {
        // sheet 设置处理的工作簿 headRowNumber 设置从excel第几行开始读取
        return EasyExcel.read(ins).build();
    }

    /**
     * 读取excel文件
     */
    public static <T> List<T> readExcel(InputStream ins, Class<T> clazz) {
        // sheet 设置处理的工作簿 headRowNumber 设置从excel第几行开始读取
        return EasyExcel.read(ins).head(clazz).sheet().headRowNumber(1).doReadSync();// 第0行一般是表头，从第1行开始读取
    }

    /**
     * 向浏览器输出excel文件
     *
     * @param response HttpServletResponse
     * @param data 输出的数据
     * @param fileName 输出的文件名称 excel的名称
     * @param sheetName 输出的excel的sheet的名称 也就是页的名称
     * @param clazz 输出数据的模板
     */
    public static <T> void WriteExcel(HttpServletResponse response, List<T> data,
            String fileName, String sheetName, Class<T> clazz) {
        HorizontalCellStyleStrategy horizontalCellStyleStrategy = getWriteHandler();
        try {
            EasyExcel.write(getOutputStream(fileName, response), clazz)
                    .excelType(ExcelTypeEnum.XLSX)
                    .sheet(sheetName)
                    .registerWriteHandler(horizontalCellStyleStrategy)
                    .doWrite(data);
        } catch (Exception e) {
            log.error("输出excel文件失败", e);
            throw new RuntimeException("输出excel文件失败", e);
        }
    }

    /**
     * 向浏览器输出excel文件
     *
     * @param data 输出的数据
     * @param fileName 输出的文件名称 excel的名称
     * @param sheetName 输出的excel的sheet的名称 也就是页的名称
     * @param clazz 输出数据的模板
     */
    public static <T> ResponseEntity<Resource> WriteExcel(List<T> data,
            String fileName, String sheetName, Class<T> clazz) {
        HorizontalCellStyleStrategy horizontalCellStyleStrategy = getWriteHandler();
        try {
            var tempFile = Files.createTempFile(null, ".xlsx").toFile();
            EasyExcel.write(tempFile, clazz)
                    .excelType(ExcelTypeEnum.XLSX)
                    .sheet(sheetName)
                    .registerWriteHandler(horizontalCellStyleStrategy)
                    .doWrite(data);
            return WriteExcel(tempFile, fileName);
        } catch (Exception e) {
            log.error("输出excel文件失败", e);
            throw new RuntimeException("输出excel文件失败", e);
        }
    }

    public static ResponseEntity<Resource> WriteExcel(File tempFile, String fileName) {
        var name = URLEncoder.encode(fileName, StandardCharsets.UTF_8);
        return ResponseEntity.ok()
                .header(HttpHeaders.CONTENT_DISPOSITION, "attachment;filename=" + name + ".xlsx")
                .header(HttpHeaders.CONTENT_TYPE,
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                .body(createTempFileResource(tempFile));

    }

    /**
     * 向浏览器输出excel文件
     *
     * @param response HttpServletResponse
     * @param sheetName 输出的文件名称 excel的名称
     * @param clazz 输出数据的模板
     */
    public static <T> void writeTemplate(HttpServletResponse response,
            String fileName, String sheetName, Class<T> clazz) {
        WriteExcel(response, Collections.emptyList(), fileName, sheetName, clazz);
    }

    public static ExcelWriter buildExcel(HttpServletResponse response, String fileName) throws IOException {
        HorizontalCellStyleStrategy horizontalCellStyleStrategy = getWriteHandler();
        return EasyExcel.write(getOutputStream(fileName, response))
                .excelType(ExcelTypeEnum.XLSX)
                .registerWriteHandler(horizontalCellStyleStrategy)
                .build();
    }

    public static ExcelWriter buildTempExcel(File tempFile) {
        HorizontalCellStyleStrategy horizontalCellStyleStrategy = getWriteHandler();
        return EasyExcel.write(tempFile)
                .excelType(ExcelTypeEnum.XLSX)
                .registerWriteHandler(horizontalCellStyleStrategy)
                .build();
    }

    public static <T> ReadSheet readSheet(
            @Nullable Integer sheetNo, @Nullable String name,
            Class<T> clazz, List<T> list) {
        return EasyExcel.readSheet(sheetNo, name)
                .head(clazz)
                .registerReadListener(new ProxyReadListener<>(list))
                .build();
    }

    public static <T> WriteSheet writerSheet(Integer sheetNo, String sheetName, Class<T> clazz,
            SheetWriteHandler... handler) {
        var builder = EasyExcel.writerSheet(sheetNo, sheetName).head(clazz);
        for (SheetWriteHandler writeHandler : handler) {
            builder.registerWriteHandler(writeHandler);
        }
        return builder.build();
    }

    public static <T> WriteSheet writerSheet(Integer sheetNo, String sheetName, Class<T> clazz) {
        var builder = EasyExcel.writerSheet(sheetNo, sheetName).head(clazz);
        return builder.build();
    }

    public static <T> WriteSheet writerSheet(String sheetName, Class<T> clazz) {
        var builder = EasyExcel.writerSheet(null, sheetName).head(clazz);
        return builder.build();
    }

    public static OutputStream getOutputStream(String fileName, HttpServletResponse response) throws IOException {
        fileName = URLEncoder.encode(fileName, StandardCharsets.UTF_8);
        response.setContentType("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet");
        response.setCharacterEncoding("utf8");
        response.setHeader("Content-Disposition", "attachment;filename=" + fileName + ".xlsx");
        return response.getOutputStream();
    }

    public static FileSystemResource createTempFileResource(File file) {
        return new TempFileResource(file);
    }

    private static final class TempFileResource extends FileSystemResource {

        private TempFileResource(File file) {
            super(file);
        }

        @NotNull
        @Override
        public ReadableByteChannel readableChannel() throws IOException {
            @SuppressWarnings("resource")
            ReadableByteChannel readableChannel = super.readableChannel();
            return new ReadableByteChannel() {

                @Override
                public boolean isOpen() {
                    return readableChannel.isOpen();
                }

                @Override
                public void close() throws IOException {
                    closeThenDeleteFile(readableChannel);
                }

                @Override
                public int read(ByteBuffer dst) throws IOException {
                    return readableChannel.read(dst);
                }
            };
        }

        @NotNull
        @Override
        public InputStream getInputStream() throws IOException {
            return new FilterInputStream(super.getInputStream()) {

                @Override
                public void close() throws IOException {
                    closeThenDeleteFile(this.in);
                }

            };
        }

        private void closeThenDeleteFile(Closeable closeable) throws IOException {
            try {
                closeable.close();
            } finally {
                deleteFile(getFile());
            }
        }

        private static void deleteFile(File file) {
            // System.out.println(file);
            file.delete();
        }

        @Override
        public boolean isFile() {
            // Prevent zero-copy so we can delete the file on close
            return false;
        }

    }

    @RequiredArgsConstructor
    private static class ProxyReadListener<T> implements ReadListener<T> {
        private final List<T> list;

        @Override
        public void invoke(T data, AnalysisContext context) {
            list.add(data);
        }

        @Override
        public void doAfterAllAnalysed(AnalysisContext context) {

        }
    }

    public static FormulaEvaluator createEvaluator(Workbook workbook) {
        return workbook.getCreationHelper()
                .createFormulaEvaluator();
    }

    public static Double getCellValueDouble(FormulaEvaluator evaluator, Cell cell) {
        switch (cell.getCellType()) {
            case NUMERIC -> {
                return cell.getNumericCellValue();
            }
            case BOOLEAN -> {
                return cell.getBooleanCellValue() ? 1.0 : 0.0;
            }
            case FORMULA -> {
                var inCell = evaluator.evaluateInCell(cell);
                return getCellValueDouble(evaluator, inCell);
            }
            case STRING -> {
                var str = cell.getStringCellValue();
                try {
                    return Double.valueOf(str);
                } catch (Exception e) {
                    return null;
                }
            }
            default -> {
                return null;
            }
        }
    }

}

