#include "vcf_encoder.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>

int
vcz_itoa(char *restrict buf, int64_t value)
{
    int p = 0;
    int j, k;

    if (value < 0) {
        buf[p] = '-';
        p++;
        value = -value;
    }
    /* We only support int32_t values. The +1 here is for supporting the
     * float converter below */
    assert(value <= (1LL + INT32_MAX));
    /*  special case small values */
    if (value < 10) {
        buf[p] = (char) value + '0';
        p++;
    } else {
        k = 0;
        if (value < 100) {
            k = 1;
        } else if (value < 1000) {
            k = 2;
        } else if (value < 10000) {
            k = 3;
        } else if (value < 100000) {
            k = 4;
        } else if (value < 1000000) {
            k = 5;
        } else if (value < 10000000) {
            k = 6;
        } else if (value < 100000000) {
            k = 7;
        } else if (value < 1000000000) {
            k = 8;
        } else if (value < 10000000000) {
            // Largest possible INT32 value
            k = 9;
        }

        // iterate backwards in buf
        p += k;
        buf[p] = (char) (value % 10) + '0';
        for (j = 0; j < k; j++) {
            p--;
            value = value / 10;
            buf[p] = (char) (value % 10) + '0';
        }
        p += k + 1;
    }
    buf[p] = '\0';
    return p;
}

int
vcz_ftoa(char *restrict buf, float value)
{
    int p = 0;
    int64_t i, d1, d2, d3;

    if (isnan(value)) {
        strcpy(buf, "nan");
        return p + 3;
    }
    if (value < 0) {
        buf[p] = '-';
        p++;
        value = -value;
    }
    if (isinf(value)) {
        strcpy(buf + p, "inf");
        return p + 3;
    }

    /* integer part */
    i = (int64_t) round(((double) value) * 1000);
    /* printf("i = %ld\n", i); */
    /* printf("i/ 1000 = %ld\n", i / 1000); */
    p += vcz_itoa(buf + p, i / 1000);

    /* fractional part */
    d3 = i % 10;
    d2 = (i / 10) % 10;
    d1 = (i / 100) % 10;
    if (d1 + d2 + d3 > 0) {
        buf[p] = '.';
        p++;
        buf[p] = (char) d1 + '0';
        p++;
        if (d2 + d3 > 0) {
            buf[p] = (char) d2 + '0';
            p++;
            if (d3 > 0) {
                buf[p] = (char) d3 + '0';
                p++;
            }
        }
    }
    buf[p] = '\0';
    return p;
}

static inline int64_t
append_string(char *restrict buf, const char *restrict str, int64_t len, int64_t offset,
    int64_t buflen)
{
    if (offset + len > buflen) {
        /* printf("offset = %d len=%d buflen = %d\n", (int) offset, (int) len, (int)
         * buflen); */
        return VCZ_ERR_BUFFER_OVERFLOW;
    }
    memcpy(buf + offset, str, (size_t) len);
    return offset + len;
}

static inline int64_t
append_char(char *restrict buf, char c, int64_t offset, int64_t buflen)
{
    if (offset == buflen) {
        return VCZ_ERR_BUFFER_OVERFLOW;
    }
    buf[offset] = c;
    return offset + 1;
}

static inline int64_t
append_int(char *restrict buf, int32_t value, int64_t offset, int64_t buflen)
{
    char tmp[VCZ_INT32_BUF_SIZE];
    int64_t len;

    if (value == VCZ_INT_MISSING) {
        return append_char(buf, '.', offset, buflen);
    }
    /* Note: it would be slightly more efficient to write directly into the output
     * buffer here, but doing it this way makes it easier to test that we're correctly
     * catching all buffer overflows */
    len = vcz_itoa(tmp, value);
    /* printf("%d: %d\n", value, (int) len); */
    return append_string(buf, tmp, len, offset, buflen);
}

static inline int64_t
append_float(
    char *restrict buf, int32_t int32_value, float value, int64_t offset, int64_t buflen)
{
    char tmp[VCZ_FLOAT32_BUF_SIZE];
    int64_t len;

    if (int32_value == VCZ_FLOAT32_MISSING_AS_INT32) {
        return append_char(buf, '.', offset, buflen);
    }
    len = vcz_ftoa(tmp, value);
    /* printf("%f: %d\n", value, (int) len); */
    return append_string(buf, tmp, len, offset, buflen);
}

static bool
bool_all_missing(const int8_t *restrict data, size_t n)
{
    assert(n == 1);
    return !data[0];
}

static bool
int32_all_missing(const int32_t *restrict data, size_t n)
{
    size_t j;

    for (j = 0; j < n; j++) {
        if (data[j] != VCZ_INT_FILL && data[j] != VCZ_INT_MISSING) {
            return false;
        }
    }
    return true;
}

static bool
float32_all_missing(const float *restrict data, size_t n)
{
    size_t j;
    const int32_t *restrict di32 = (const int32_t *) data;

    for (j = 0; j < n; j++) {
        if (di32[j] != VCZ_FLOAT32_FILL_AS_INT32
            && di32[j] != VCZ_FLOAT32_MISSING_AS_INT32) {
            return false;
        }
    }
    return true;
}
static bool
string_all_missing(const char *restrict data, size_t item_size, size_t n)
{
    size_t j;

    for (j = 0; j < n * item_size; j++) {
        if (data[j] != VCZ_STRING_FILL && data[j] != VCZ_STRING_MISSING) {
            return false;
        }
    }
    return true;
}

static int64_t
string_field_write_entry(
    const vcz_field_t *self, const void *data, char *buf, int64_t buflen, int64_t offset)
{
    const char *source = (const char *) data;
    size_t column, byte;
    const char sep = ',';
    int64_t source_offset = 0;

    for (column = 0; column < self->num_columns; column++) {
        if (column > 0 && source[source_offset] != '\0') {
            offset = append_char(buf, sep, offset, buflen);
            if (offset < 0) {
                goto out;
            }
        }
        for (byte = 0; byte < self->item_size; byte++) {
            if (source[source_offset] != '\0') {
                offset = append_char(buf, source[source_offset], offset, buflen);
                if (offset < 0) {
                    goto out;
                }
            }
            source_offset++;
        }
    }
out:
    return offset;
}

static int64_t
bool_field_write_entry(const vcz_field_t *VCZ_UNUSED(self), const void *VCZ_UNUSED(data),
    char *VCZ_UNUSED(buf), size_t VCZ_UNUSED(buflen), int64_t offset)
{
    /* Erase the previously inserted "=" we need for other fields.*/
    return offset - 1;
}

static int64_t
int32_field_write_entry(const vcz_field_t *self, const void *restrict data, char *buf,
    int64_t buflen, int64_t offset, char separator)
{
    const int32_t *restrict source = (const int32_t *) data;
    size_t column;

    for (column = 0; column < self->num_columns; column++) {
        if (source[column] == VCZ_INT_FILL) {
            break;
        }
        if (column > 0) {
            offset = append_char(buf, separator, offset, buflen);
            if (offset < 0) {
                goto out;
            }
        }
        offset = append_int(buf, source[column], offset, buflen);
        if (offset < 0) {
            goto out;
        }
    }
out:
    return offset;
}

static int64_t
float32_field_write_entry(const vcz_field_t *self, const void *restrict data,
    char *restrict buf, int64_t buflen, int64_t offset)
{
    const float *restrict source = (const float *restrict) data;
    const int32_t *restrict int32_source = (const int32_t *restrict) data;
    int32_t int32_value;
    size_t column;

    for (column = 0; column < self->num_columns; column++) {
        int32_value = int32_source[column];
        if (int32_value == VCZ_FLOAT32_FILL_AS_INT32) {
            break;
        }
        if (column > 0) {
            offset = append_char(buf, ',', offset, buflen);
            if (offset < 0) {
                goto out;
            }
        }
        offset = append_float(buf, int32_value, source[column], offset, buflen);
        if (offset < 0) {
            goto out;
        }
    }
out:
    return offset;
}

static int64_t
vcz_field_write_entry(
    const vcz_field_t *self, const void *data, char *buf, int64_t buflen, int64_t offset)
{
    if (self->type == VCZ_TYPE_INT) {
        assert(self->item_size == 4);
        return int32_field_write_entry(self, data, buf, buflen, offset, ',');
    } else if (self->type == VCZ_TYPE_FLOAT) {
        assert(self->item_size == 4);
        return float32_field_write_entry(self, data, buf, buflen, offset);
    } else if (self->type == VCZ_TYPE_BOOL) {
        assert(self->item_size == 1);
        assert(self->num_columns == 1);
        return bool_field_write_entry(self, data, buf, (size_t) buflen, offset);
    }
    assert(self->type == VCZ_TYPE_STRING);
    return string_field_write_entry(self, data, buf, buflen, offset);
}

int64_t
vcz_field_write(
    const vcz_field_t *self, size_t variant, char *buf, int64_t buflen, int64_t offset)
{

    size_t row_size = self->num_columns * self->item_size;
    const void *data = self->data + variant * row_size;

    return vcz_field_write_entry(self, data, buf, buflen, offset);
}

static bool
vcz_info_field_is_missing(const vcz_field_t *self, size_t variant)
{
    size_t row_size = self->num_columns * self->item_size;
    const void *data = self->data + variant * row_size;

    if (self->type == VCZ_TYPE_INT) {
        if (self->item_size == 4) {
            return int32_all_missing(data, self->num_columns);
        }
    } else if (self->type == VCZ_TYPE_FLOAT) {
        return float32_all_missing(data, self->num_columns);
    } else if (self->type == VCZ_TYPE_BOOL) {
        return bool_all_missing(data, self->num_columns);
    } else if (self->type == VCZ_TYPE_STRING) {
        return string_all_missing(data, self->item_size, self->num_columns);
    }
    assert(false);
    return false;
}

static bool
vcz_format_field_is_missing(const vcz_field_t *self, size_t variant, size_t num_samples)
{
    size_t row_size = self->num_columns * self->item_size * num_samples;
    const void *data = self->data + variant * row_size;

    if (self->type == VCZ_TYPE_INT) {
        if (self->item_size == 4) {
            return int32_all_missing(data, self->num_columns * num_samples);
        }
    } else if (self->type == VCZ_TYPE_FLOAT) {
        return float32_all_missing(data, self->num_columns * num_samples);
    } else if (self->type == VCZ_TYPE_STRING) {
        return string_all_missing(
            data, self->item_size, self->num_columns * num_samples);
    }
    assert(false);
    return false;
}

void
vcz_field_print_state(const vcz_field_t *self, FILE *out)
{
    fprintf(out, "\t%s\ttype:%d\titem_size=%d\tnum_columns=%d\tdata=%p\n", self->name,
        self->type, (int) self->item_size, (int) self->num_columns, self->data);
}

static int
vcz_field_init(vcz_field_t *self, const char *name, int type, size_t item_size,
    size_t num_columns, const void *data)
{
    int ret = 0;

    self->name_length = strlen(name);
    if (self->name_length >= VCZ_MAX_FIELD_NAME_LEN) {
        ret = VCZ_ERR_FIELD_NAME_TOO_LONG;
        goto out;
    }
    // TODO MORE CHECKS
    if (type == VCZ_TYPE_BOOL && item_size != 1) {
        ret = VCZ_ERR_FIELD_UNSUPPORTED_TYPE;
        goto out;
    }
    strcpy(self->name, name);
    self->type = type;
    self->item_size = item_size;
    self->num_columns = num_columns;
    self->data = data;
out:
    return ret;
}

static int64_t
vcz_variant_encoder_write_sample_gt(const vcz_variant_encoder_t *self, size_t variant,
    size_t sample, char *buf, int64_t buflen, int64_t offset)
{
    const size_t ploidy = self->gt.num_columns;
    size_t source_offset = variant * self->num_samples * ploidy + sample * ploidy;
    const int32_t *source = ((const int32_t *) self->gt.data) + source_offset;
    const bool phased = self->gt_phased_data[variant * self->num_samples + sample];
    char sep = phased ? '|' : '/';

    return int32_field_write_entry(&self->gt, source, buf, buflen, offset, sep);
}

static int64_t
vcz_variant_encoder_write_info_fields(const vcz_variant_encoder_t *self, size_t variant,
    char *buf, int64_t buflen, int64_t offset)
{
    vcz_field_t field;
    size_t j;
    bool *missing = NULL;
    bool all_missing = true;
    bool first_field;

    if (self->num_info_fields > 0) {
        missing = malloc(self->num_info_fields * sizeof(*missing));
        if (missing == NULL) {
            offset = VCZ_ERR_NO_MEMORY;
            goto out;
        }
        for (j = 0; j < self->num_info_fields; j++) {
            missing[j] = vcz_info_field_is_missing(&self->info_fields[j], variant);
            if (!missing[j]) {
                all_missing = false;
            }
        }
    }

    if (all_missing) {
        offset = append_string(buf, ".\t", 2, offset, buflen);
        if (offset < 0) {
            goto out;
        }
    } else {
        first_field = true;
        for (j = 0; j < self->num_info_fields; j++) {
            if (!missing[j]) {
                if (!first_field) {
                    buf[offset - 1] = ';';
                }
                first_field = false;
                field = self->info_fields[j];
                offset = append_string(
                    buf, field.name, (int64_t) field.name_length, offset, buflen);
                if (offset < 0) {
                    goto out;
                }
                offset = append_char(buf, '=', offset, buflen);
                if (offset < 0) {
                    goto out;
                }
                offset = vcz_field_write(&field, variant, buf, buflen, offset);
                if (offset < 0) {
                    goto out;
                }
                offset = append_char(buf, '\t', offset, buflen);
                if (offset < 0) {
                    goto out;
                }
            }
        }
    }
out:
    if (missing != NULL) {
        free(missing);
    }
    return offset;
}

static int64_t
vcz_variant_encoder_write_format_fields(const vcz_variant_encoder_t *self,
    size_t variant, char *buf, int64_t buflen, int64_t offset)
{
    size_t j, sample, row_size;
    vcz_field_t field;
    bool *missing = NULL;
    bool all_missing = true;
    bool has_gt = (self->gt.data != NULL);
    bool gt_missing = true;
    const size_t num_samples = self->num_samples;
    const void *data;

    if (has_gt) {
        gt_missing = vcz_format_field_is_missing(&self->gt, variant, num_samples);
    }

    if (self->num_format_fields > 0) {
        missing = malloc(self->num_format_fields * sizeof(*missing));
        if (missing == NULL) {
            offset = VCZ_ERR_NO_MEMORY;
            goto out;
        }
        for (j = 0; j < self->num_format_fields; j++) {
            missing[j] = vcz_format_field_is_missing(
                &self->format_fields[j], variant, num_samples);
            if (!missing[j]) {
                all_missing = false;
            }
        }
    }
    all_missing = all_missing && gt_missing;

    if (all_missing) {
        for (j = 0; j < num_samples + 1; j++) {
            offset = append_string(buf, ".\t", 2, offset, buflen);
            if (offset < 0) {
                goto out;
            }
        }
    } else {
        if (!gt_missing) {
            offset = append_string(buf, "GT:", 3, offset, buflen);
            if (offset < 0) {
                goto out;
            }
        }
        for (j = 0; j < self->num_format_fields; j++) {
            if (!missing[j]) {
                field = self->format_fields[j];
                offset = append_string(
                    buf, field.name, (int64_t) field.name_length, offset, buflen);
                if (offset < 0) {
                    goto out;
                }
                offset = append_char(buf, ':', offset, buflen);
                if (offset < 0) {
                    goto out;
                }
            }
        }
        buf[offset - 1] = '\t';

        for (sample = 0; sample < num_samples; sample++) {
            if (!gt_missing) {
                offset = vcz_variant_encoder_write_sample_gt(
                    self, variant, sample, buf, buflen, offset);
                if (offset < 0) {
                    goto out;
                }
                offset = append_char(buf, ':', offset, buflen);
                if (offset < 0) {
                    goto out;
                }
            }
            for (j = 0; j < self->num_format_fields; j++) {
                if (!missing[j]) {
                    field = self->format_fields[j];
                    row_size = num_samples * field.num_columns * field.item_size;
                    data = field.data + variant * row_size
                           + sample * field.num_columns * field.item_size;
                    offset = vcz_field_write_entry(&field, data, buf, buflen, offset);
                    if (offset < 0) {
                        goto out;
                    }
                    offset = append_char(buf, ':', offset, buflen);
                    if (offset < 0) {
                        goto out;
                    }
                }
            }
            buf[offset - 1] = '\t';
        }
    }
out:
    if (missing != NULL) {
        free(missing);
    }
    return offset;
}

static int64_t
vcz_variant_encoder_write_filter(const vcz_variant_encoder_t *self, size_t variant,
    char *buf, int64_t buflen, int64_t offset)
{
    const vcz_field_t filter_id = self->filter_id;
    bool all_missing = true;
    const int8_t *restrict data = self->filter_data + (variant * filter_id.num_columns);
    const char *filter_id_data = (const char *) self->filter_id.data;
    size_t j, k, source_offset;

    for (j = 0; j < filter_id.num_columns; j++) {
        if (data[j]) {
            all_missing = false;
        }
    }
    if (all_missing) {
        offset = append_char(buf, '.', offset, buflen);
        if (offset < 0) {
            goto out;
        }
    } else {
        source_offset = 0;
        for (j = 0; j < filter_id.num_columns; j++) {
            if (data[j]) {
                source_offset = j * filter_id.item_size;
                for (k = 0; k < filter_id.item_size; k++) {
                    if (filter_id_data[source_offset] == VCZ_STRING_FILL) {
                        break;
                    }
                    offset = append_char(
                        buf, filter_id_data[source_offset], offset, buflen);
                    if (offset < 0) {
                        goto out;
                    }
                    source_offset++;
                }
            }
        }
    }
    offset = append_char(buf, '\t', offset, buflen);
    if (offset < 0) {
        goto out;
    }
out:
    return offset;
}

int64_t
vcz_variant_encoder_write_row(
    const vcz_variant_encoder_t *self, size_t variant, char *buf, size_t buflen)
{
    int64_t offset = 0;
    size_t j;

    for (j = 0; j < VCZ_NUM_FIXED_FIELDS; j++) {
        if (vcz_info_field_is_missing(&self->fixed_fields[j], variant)) {
            offset = append_char(buf, '.', offset, (int64_t) buflen);
            if (offset < 0) {
                goto out;
            }
        } else {
            offset = vcz_field_write(
                &self->fixed_fields[j], variant, buf, (int64_t) buflen, offset);
            if (offset < 0) {
                goto out;
            }
        }
        offset = append_char(buf, '\t', offset, (int64_t) buflen);
        if (offset < 0) {
            goto out;
        }
    }
    offset
        = vcz_variant_encoder_write_filter(self, variant, buf, (int64_t) buflen, offset);
    if (offset < 0) {
        goto out;
    }
    offset = vcz_variant_encoder_write_info_fields(
        self, variant, buf, (int64_t) buflen, offset);
    if (offset < 0) {
        goto out;
    }
    offset = vcz_variant_encoder_write_format_fields(
        self, variant, buf, (int64_t) buflen, offset);
    if (offset < 0) {
        goto out;
    }
    offset--;
    buf[offset] = '\0';
out:
    return offset;
}

void
vcz_variant_encoder_print_state(const vcz_variant_encoder_t *self, FILE *out)
{
    size_t j;

    fprintf(out, "vcz_variant_encoder: %p\n", (const void *) self);
    fprintf(out, "\tnum_samples: %d\n", (int) self->num_samples);
    fprintf(out, "\tnum_variants: %d\n", (int) self->num_variants);
    vcz_field_print_state(&self->filter_id, out);
    for (j = 0; j < VCZ_NUM_FIXED_FIELDS; j++) {
        vcz_field_print_state(&self->fixed_fields[j], out);
    }
    fprintf(out, "\tINFO:\n");
    for (j = 0; j < self->num_info_fields; j++) {
        vcz_field_print_state(&self->info_fields[j], out);
    }
    fprintf(out, "\tFORMAT:\n");
    if (self->gt.data != NULL) {
        vcz_field_print_state(&self->gt, out);
    }
    for (j = 0; j < self->num_format_fields; j++) {
        vcz_field_print_state(&self->format_fields[j], out);
    }
}

int
vcz_variant_encoder_add_info_field(vcz_variant_encoder_t *self, const char *name,
    int type, size_t item_size, size_t num_columns, const void *data)
{
    int ret = 0;
    vcz_field_t *tmp, *field;

    if (self->num_info_fields == self->max_info_fields) {
        self->max_info_fields += self->field_array_size_increment;
        tmp = realloc(
            self->info_fields, self->max_info_fields * sizeof(*self->info_fields));
        if (tmp == NULL) {
            ret = VCZ_ERR_NO_MEMORY;
            goto out;
        }
        self->info_fields = tmp;
    }
    field = self->info_fields + self->num_info_fields;
    ret = vcz_field_init(field, name, type, item_size, num_columns, data);
    if (ret != 0) {
        goto out;
    }
    self->num_info_fields++;
out:
    return ret;
}

int
vcz_variant_encoder_add_format_field(vcz_variant_encoder_t *self, const char *name,
    int type, size_t item_size, size_t num_columns, const void *data)
{
    int ret = 0;
    vcz_field_t *tmp, *field;

    if (self->num_format_fields == self->max_format_fields) {
        self->max_format_fields += self->field_array_size_increment;
        /* NOTE: assuming realloc(NULL) is safe and portable. check */
        tmp = realloc(
            self->format_fields, self->max_format_fields * sizeof(*self->format_fields));
        if (tmp == NULL) {
            ret = VCZ_ERR_NO_MEMORY;
            goto out;
        }
        self->format_fields = tmp;
    }
    field = self->format_fields + self->num_format_fields;
    self->num_format_fields++;

    ret = vcz_field_init(field, name, type, item_size, num_columns, data);
out:
    return ret;
}

int
vcz_variant_encoder_add_gt_field(vcz_variant_encoder_t *self, size_t item_size,
    size_t num_columns, const void *data, const int8_t *phased_data)
{
    self->gt_phased_data = phased_data;
    return vcz_field_init(&self->gt, "GT", VCZ_TYPE_INT, item_size, num_columns, data);
}

#define CHROM_FIELD_INDEX 0
#define POS_FIELD_INDEX 1
#define ID_FIELD_INDEX 2
#define REF_FIELD_INDEX 3
#define ALT_FIELD_INDEX 4
#define QUAL_FIELD_INDEX 5
#define FILTER_FIELD_INDEX 6

int
vcz_variant_encoder_add_chrom_field(
    vcz_variant_encoder_t *self, size_t item_size, const char *data)
{
    return vcz_field_init(self->fixed_fields + CHROM_FIELD_INDEX, "CHROM",
        VCZ_TYPE_STRING, item_size, 1, data);
}

int
vcz_variant_encoder_add_pos_field(vcz_variant_encoder_t *self, const int32_t *data)
{
    return vcz_field_init(
        self->fixed_fields + POS_FIELD_INDEX, "POS", VCZ_TYPE_INT, 4, 1, data);
}

int
vcz_variant_encoder_add_qual_field(vcz_variant_encoder_t *self, const float *data)
{
    return vcz_field_init(
        self->fixed_fields + QUAL_FIELD_INDEX, "QUAL", VCZ_TYPE_FLOAT, 4, 1, data);
}

int
vcz_variant_encoder_add_ref_field(
    vcz_variant_encoder_t *self, size_t item_size, const char *data)
{
    return vcz_field_init(self->fixed_fields + REF_FIELD_INDEX, "REF", VCZ_TYPE_STRING,
        item_size, 1, data);
}

int
vcz_variant_encoder_add_id_field(
    vcz_variant_encoder_t *self, size_t item_size, size_t num_columns, const char *data)
{
    return vcz_field_init(self->fixed_fields + ID_FIELD_INDEX, "ID", VCZ_TYPE_STRING,
        item_size, num_columns, data);
}

int
vcz_variant_encoder_add_alt_field(
    vcz_variant_encoder_t *self, size_t item_size, size_t num_columns, const char *data)
{
    return vcz_field_init(self->fixed_fields + ALT_FIELD_INDEX, "ALT", VCZ_TYPE_STRING,
        item_size, num_columns, data);
}

int
vcz_variant_encoder_add_filter_field(vcz_variant_encoder_t *self, size_t id_item_size,
    size_t id_num_columns, const char *id_data, const int8_t *filter_data)
{
    self->filter_data = filter_data;
    return vcz_field_init(&self->filter_id, "IDS/FILTERS", VCZ_TYPE_STRING, id_item_size,
        id_num_columns, id_data);
}

int
vcz_variant_encoder_init(
    vcz_variant_encoder_t *self, size_t num_variants, size_t num_samples)
{
    memset(self, 0, sizeof(*self));
    self->num_samples = num_samples;
    self->num_variants = num_variants;
    self->field_array_size_increment = 64; // arbitrary
    return 0;
}

void
vcz_variant_encoder_free(vcz_variant_encoder_t *self)
{
    if (self->info_fields != NULL) {
        free(self->info_fields);
        self->info_fields = NULL;
    }
    if (self->format_fields != NULL) {
        free(self->format_fields);
        self->format_fields = NULL;
    }
}
