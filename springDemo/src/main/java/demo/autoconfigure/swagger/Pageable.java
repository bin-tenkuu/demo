
package demo.autoconfigure.swagger;

import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Min;
import lombok.Data;
import org.springdoc.core.annotations.ParameterObject;

/**
 * The type Pageable.
 *
 * @author bnasslahsen
 */
@Data
@ParameterObject
public class Pageable {

    /**
     * The Page.
     */
    @Min(0)
    @Parameter(description = "页码 (1..N)", schema = @Schema(type = "integer", defaultValue = "1"))
    private final Integer page;

    /**
     * The Size.
     */
    @Min(1)
    @Parameter(description = "每页数量", schema = @Schema(type = "integer", defaultValue = "5"))
    private final Integer size;
}
