package demo.util;

import jakarta.validation.ConstraintViolation;
import jakarta.validation.Validator;
import org.springframework.validation.BindException;
import org.springframework.validation.DirectFieldBindingResult;
import org.springframework.validation.FieldError;

import java.util.Set;

/// bean对象属性验证
@SuppressWarnings("unused")
public class BeanValidators {
    public static void validateWithException(Validator validator, Object object) throws BindException {
        Set<ConstraintViolation<Object>> constraintViolations = validator.validate(object);
        if (constraintViolations.isEmpty()) {
            return;
        }
        var objectName = object.getClass().getSimpleName();
        var result = new DirectFieldBindingResult(object, objectName);
        for (ConstraintViolation<Object> violation : constraintViolations) {
            var fieldError = new FieldError(
                    objectName, violation.getPropertyPath().toString(),
                    violation.getMessage()
            );
            result.addError(fieldError);
        }
        throw new BindException(result);
    }
}
