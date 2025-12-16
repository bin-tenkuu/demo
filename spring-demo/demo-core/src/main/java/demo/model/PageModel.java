package demo.model;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

/// @author bin
/// @since 2023/01/31
@SuppressWarnings("unused")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Schema(description = "分页")
public class PageModel {
    @Schema(description = "当前页", example = "1")
    private int currentPage = 1;
    @Schema(description = "每页数量", example = "5")
    private int pageSize = 5;
    @Schema(description = "总页数", example = "20")
    private int totalPage = 1;
    @Schema(description = "总数", example = "99")
    private long totalCount = 0L;
}
