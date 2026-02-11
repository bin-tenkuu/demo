package demo.core.model;

import demo.core.enums.TimeType;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.RequiredArgsConstructor;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;

/// @author bin
/// @version 1.0.0
/// @since 2024/11/01
@SuppressWarnings("unused")
@Data
@Schema(description = "图表数据")
@RequiredArgsConstructor
public class ChartModel {
    @Schema(description = "标题")
    private final String chartName;
    @Schema(description = "横坐标")
    private List<String> xs;
    @Schema(description = "内容")
    private List<ChartData> labels = new ArrayList<>();

    public void setXsTime(TimeType type, List<LocalDateTime> xs) {
        this.setXs(type.toString(xs));
    }

    public void add(ChartData chartData) {
        labels.add(chartData);
    }

    public void add(String name, Collection<?> data) {
        labels.add(new ChartData(name, new ArrayList<>(data)));
    }

    public void add(Map<String, ? extends Collection<?>> map) {
        for (var entry : map.entrySet()) {
            add(entry.getKey(), entry.getValue());
        }
    }

    public void add(String name, double scale) {
        var list = new ArrayList<Number>(xs.size());
        for (int i = 0; i < xs.size(); i++) {
            list.add(Math.random() * scale);
        }
        labels.add(new ChartData(name, list));
    }

    @Data
    @RequiredArgsConstructor
    public static class ChartData {
        @Schema(description = "标签")
        private final String name;
        @Schema(description = "数据")
        private final List<?> data;
    }
}
